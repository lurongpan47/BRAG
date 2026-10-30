"""
B-RAG: Bodhi-Regularised Alignment Gradient.

Proximal policy optimisation (clipped surrogate, GAE, tabular softmax policy)
of the Bodhi-regularised objective

    J_lambda(pi) = E[ sum_t gamma^t ( r_t - lambda (alpha d_K(s_{t+1}) + beta d_A(s_{t+1})) ) ]
    subject to  C(pi) = E[ sum_t gamma^t h_t ] <= eps,

with the Metta constraint handled either by dual ascent on a Lagrange multiplier
(PPO-Lagrangian) or by an interior-point log-barrier whose gradient is an
adaptive multiplier nu = 1/(t (eps - C)) (IPO).  The constraint cost of each
iterate is evaluated exactly from the model (tabular setting); all reported
numbers are exact dynamic-programming quantities, never Monte-Carlo estimates.

Lojong (mind-training) protocol options:
  init      'uniform' | 'noop' (positive seed: start from the inaction policy) | 'random'
  schedule  'early' (regularisation from the first iteration) | 'late' (task-only, then switched on)
  slip_train list of slip probabilities drawn per iteration (adversity randomisation);
            the robust constraint uses the worst case over the list
The critics (d_K, d_A, h) are frozen: they are computed from the model and
never updated on the agent's own trajectories.
"""
from __future__ import annotations

import numpy as np

from env import BodhiGridworld, NOOP
from critics import Evaluator, penalty_tables


def softmax(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class Adam:
    def __init__(self, shape, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.m, self.v, self.t = np.zeros(shape), np.zeros(shape), 0
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps

    def step(self, g, lr_scale=1.0):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        return self.lr * lr_scale * mhat / (np.sqrt(vhat) + self.eps)


class BRAG:
    def __init__(self, env: BodhiGridworld, dK, dA, *, lam=1.0, alpha=1.0, beta=1.0, metta='barrier', eps=0.1,
                 penalty='level', init='uniform', schedule='early', switch_iter=150, slip_train=(0.0,),
                 iters=300, n_envs=32, horizon=64, lr=0.05, clip=0.2, epochs=4, minibatch=256,
                 ent_coef=0.01, gae_lambda=0.95, v_lr=0.5, dual_lr=1.0, nu_max=20.0,
                 barrier_t0=5.0, barrier_t1=100.0, shield=True, shield_tau=0.5, use_preliminaries=True, pre_tol=0.1, pre_ent=0.05, ent_hi=0.1, anneal_frac=0.5, v_init=1.0,
                 seed=0, log_every=1, eval_env=None, verbose=False):
        self.env, self.dK, self.dA = env, dK, dA
        self.lam, self.alpha, self.beta, self.metta, self.eps = lam, alpha, beta, metta, eps
        self.penalty, self.init, self.schedule, self.switch_iter = penalty, init, schedule, switch_iter
        self.iters, self.n_envs, self.horizon = iters, n_envs, horizon
        self.lr, self.clip, self.epochs, self.minibatch = lr, clip, epochs, minibatch
        self.ent_coef, self.gae_lambda, self.v_lr = ent_coef, gae_lambda, v_lr
        self.dual_lr, self.nu_max, self.t0, self.t1 = dual_lr, nu_max, barrier_t0, barrier_t1
        self.use_preliminaries, self.pre_tol, self.pre_ent = use_preliminaries, pre_tol, pre_ent
        self.ent_hi, self.anneal_frac, self.v_init = ent_hi, anneal_frac, v_init
        self.rng = np.random.default_rng(seed)
        self.verbose, self.log_every = verbose, log_every
        # training environments (adversity randomisation) and exact evaluators
        self.train_envs = [env if p == env.slip else BodhiGridworld(slip=p, gamma=env.gamma, q=env.q)
                           for p in slip_train]
        self.eval_env = eval_env or env
        self.evaluator = Evaluator(self.eval_env, dK, dA, lam, alpha, beta, penalty)
        self.cost_evals = [Evaluator(e, dK, dA, lam, alpha, beta, penalty) for e in self.train_envs]
        # per-transition penalty d(s,s') (already scaled by lambda)
        _, self.pen_trans = penalty_tables(env, dK, dA, lam, alpha, beta, penalty)
        # policy and critics
        S, A = env.ns, env.na
        if init == 'uniform':
            self.logits = np.zeros((S, A))
        elif init == 'noop':
            self.logits = np.zeros((S, A)); self.logits[:, NOOP] = 2.0
        elif init == 'random':
            self.logits = self.rng.normal(0, 1.0, (S, A))
        else:
            raise ValueError(init)
        # pre-action shield (part of the Metta term): block actions whose *model-predicted* one-step
        # increase in suffering exceeds shield_tau (the flood, not the trampling)
        self.mask = np.zeros((S, A))
        self.shield_on = metta != 'none' and shield
        if self.shield_on:
            from env import H_FLOOD
            # suffering newly caused by the step under the nominal model: trampling + change in flood level
            new_suffering = env.COST - H_FLOOD * env.dam[:, None]
            self.mask[(new_suffering > shield_tau) & env.alive[:, None]] = -1e9
        self.V = np.full(S, v_init); self.Vc = np.zeros(S)      # optimistic task critic
        self.opt = Adam((S, A), lr)
        self.nu = 0.0
        self.history = []

    # --------------------------------------------------------------------- rollouts
    def rollout(self, env, pi):
        """Collect n_envs x horizon transitions.  Returns (S, A, R_task, PEN, C, S2), each (T, N)."""
        rng, N, T = self.rng, self.n_envs, self.horizon
        s = np.full(N, env.s0)
        S_, A_, R_, P_, C_, S2_ = [], [], [], [], [], []
        for t in range(T):
            u = rng.random(N)
            a = (pi[s].cumsum(axis=1) > u[:, None]).argmax(axis=1)
            s2, r, c = env.sample_step(s, a, rng)
            S_.append(s); A_.append(a); R_.append(r); P_.append(self.pen_trans[s, s2]); C_.append(c); S2_.append(s2)
            s = s2
        return tuple(np.stack(x) for x in (S_, A_, R_, P_, C_, S2_))

    def gae(self, Vtab, S, R, S2):
        """Generalised advantage estimation on a (T, N) batch with bootstrapping from V(S2)."""
        g, lam_g = self.env.gamma, self.gae_lambda
        T = S.shape[0]
        adv = np.zeros_like(R); last = np.zeros(R.shape[1])
        for t in reversed(range(T)):
            delta = R[t] + g * Vtab[S2[t]] - Vtab[S[t]]
            last = delta + g * lam_g * last
            adv[t] = last
        ret = adv + Vtab[S]
        return adv, ret

    def fit_values(self, Vtab, S, ret):
        """Tabular value regression: move each visited state's value toward its batch-mean target."""
        s_flat, ret_flat = S.ravel(), ret.ravel()
        sums = np.bincount(s_flat, weights=ret_flat, minlength=self.env.ns)
        cnts = np.bincount(s_flat, minlength=self.env.ns)
        vis = cnts > 0
        Vtab[vis] += self.v_lr * (sums[vis] / cnts[vis] - Vtab[vis])

    # --------------------------------------------------------------------- PPO update
    def ppo_update(self, logits0, S, A, adv, lr_scale=1.0, ent_coef=None):
        """Clipped-surrogate PPO epochs on tabular softmax logits; returns new logits."""
        ent_coef = self.ent_coef if ent_coef is None else ent_coef
        S, A, adv = S.ravel(), A.ravel(), adv.ravel()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(S)
        mask = self.mask
        pi0 = softmax(logits0 + mask)
        logp_old = np.log(pi0[S, A] + 1e-12)
        logits = logits0.copy()
        opt = Adam(logits.shape, self.lr); opt.m, opt.v, opt.t = self.opt.m.copy(), self.opt.v.copy(), self.opt.t
        for _ in range(self.epochs):
            perm = self.rng.permutation(n)
            for i in range(0, n, self.minibatch):
                idx = perm[i:i + self.minibatch]
                s, a, ad = S[idx], A[idx], adv[idx]
                pi = softmax(logits + mask)
                logp = np.log(pi[s, a] + 1e-12)
                ratio = np.exp(logp - logp_old[idx])
                # gradient of min(ratio*A, clip(ratio)*A): zero where the clipped branch is active
                active = ~(((ad > 0) & (ratio > 1 + self.clip)) | ((ad < 0) & (ratio < 1 - self.clip)))
                coef = np.where(active, ratio * ad, 0.0) / len(idx)
                grad = np.zeros_like(logits)
                onehot = np.zeros((len(idx), logits.shape[1])); onehot[np.arange(len(idx)), a] = 1.0
                np.add.at(grad, s, coef[:, None] * (onehot - pi[s]))
                # entropy bonus: dH/dtheta_b = -pi_b (log pi_b + H)
                ps = pi[s]; Hs = -(ps * np.log(ps + 1e-12)).sum(axis=1, keepdims=True)
                np.add.at(grad, s, -ent_coef * ps * (np.log(ps + 1e-12) + Hs) / len(idx))
                logits += opt.step(grad, lr_scale)          # gradient ascent
        self.opt.m, self.opt.v, self.opt.t = opt.m, opt.v, opt.t
        logits = logits - logits.mean(axis=1, keepdims=True)
        return logits

    def policy(self, logits=None):
        return softmax((self.logits if logits is None else logits) + self.mask)

    # --------------------------------------------------------------------- exact bookkeeping
    def robust_cost(self, pi):
        return max(ev.cost(pi) for ev in self.cost_evals)

    def penalty_level(self, pi):
        """Exact expected discounted soft penalty D(pi) = J_task(pi) - J_reg(pi) >= 0 (nominal world)."""
        _, V_task, V_reg, _ = self.evaluator.values(pi)
        return float(V_task[self.env.s0] - V_reg[self.env.s0])

    def multiplier(self, Ck, k):
        if self.metta == 'dual':
            return float(np.clip(self.nu + self.dual_lr * (Ck - self.eps), 0.0, self.nu_max))
        if self.metta == 'barrier':
            t_b = self.t0 * (self.t1 / self.t0) ** (k / max(self.iters - 1, 1))
            return 1.0 / (t_b * (self.eps - Ck)) if Ck < self.eps else self.nu_max
        return 0.0

    # --------------------------------------------------------------------- lojong: preliminaries
    def preliminaries(self, max_iters=150):
        """Before the task is pursued, learn a harmless *exploratory* disposition: minimise the
        regularisation penalty and the constraint cost alone, with a high entropy bonus, until the exact
        penalty level D(pi) < pre_tol and the constraint holds with margin.  This is the 'positive seed'
        from which the main practice starts; it also serves as the interior-point feasibility phase."""
        n = 0
        while n < max_iters:
            pi = self.policy()
            D, Cr = self.penalty_level(pi), self.robust_cost(pi)
            if D < self.pre_tol and Cr < 0.8 * self.eps:
                break
            env = self.train_envs[self.rng.integers(len(self.train_envs))]
            S, A, R, PEN, C, S2 = self.rollout(env, pi)
            advp, retp = self.gae(self.V, S, -PEN, S2)
            advc, retc = self.gae(self.Vc, S, C, S2)
            self.fit_values(self.V, S, retp); self.fit_values(self.Vc, S, retc)
            nu = (1.0 / (self.t0 * (self.eps - Cr))) if (self.metta != 'none' and Cr < self.eps) else (self.nu_max if self.metta != 'none' else 0.0)
            self.logits = self.ppo_update(self.logits, S, A, advp - nu * advc, ent_coef=self.pre_ent)
            n += 1
        self.V[:] = self.v_init    # the task critic starts fresh (optimistic) for the main practice
        return n

    # --------------------------------------------------------------------- main loop
    def train(self):
        self.pre_iters, self.feasible_from_start = 0, None
        for k in range(self.iters):
            reg_on = (self.schedule == 'early') or (k >= self.switch_iter)
            if reg_on and (k == 0 or (self.schedule == 'late' and k == self.switch_iter)) and (self.lam > 0 or self.metta != 'none'):
                self.feasible_from_start = self.robust_cost(self.policy()) < self.eps
                if self.use_preliminaries:
                    self.pre_iters = self.preliminaries()
            env = self.train_envs[self.rng.integers(len(self.train_envs))]
            pi = self.policy()
            S, A, R, PEN, C, S2 = self.rollout(env, pi)
            Rs = R - PEN if reg_on else R
            adv, ret = self.gae(self.V, S, Rs, S2)
            advc, retc = self.gae(self.Vc, S, C, S2)
            self.fit_values(self.V, S, ret); self.fit_values(self.Vc, S, retc)
            self.nu = self.multiplier(self.robust_cost(pi), k) if (reg_on and self.metta != 'none') else 0.0
            comb = adv - self.nu * advc
            # exploration schedule: entropy coefficient decays geometrically from ent_hi to ent_coef
            frac = min(1.0, k / max(1, self.anneal_frac * self.iters))
            ent_k = self.ent_hi * (self.ent_coef / self.ent_hi) ** frac
            new_logits = self.ppo_update(self.logits, S, A, comb, ent_coef=ent_k)
            if reg_on and self.metta == 'barrier':
                # interior-point safeguard: never accept an infeasible iterate
                scale, tries = 0.5, 0
                m, v, t = self.opt.m.copy(), self.opt.v.copy(), self.opt.t
                while self.robust_cost(self.policy(new_logits)) >= self.eps and tries < 6:
                    self.opt.m, self.opt.v, self.opt.t = m, v, t
                    new_logits = self.ppo_update(self.logits, S, A, comb, lr_scale=scale, ent_coef=ent_k)
                    scale *= 0.5; tries += 1
                if self.robust_cost(self.policy(new_logits)) >= self.eps:
                    self.opt.m, self.opt.v, self.opt.t = m, v, t
                    new_logits = self.ppo_update(self.logits, S, A, -advc)      # recovery step
                    if self.robust_cost(self.policy(new_logits)) >= self.eps:
                        new_logits = self.logits                                # stay put
            self.logits = new_logits
            if k % self.log_every == 0 or k == self.iters - 1:
                rec = self.evaluator.evaluate(self.policy())
                rec.update({'iter': k, 'nu': self.nu, 'reg_on': bool(reg_on)})
                self.history.append(rec)
                if self.verbose and k % 25 == 0:
                    print(f"it {k:4d} J={rec['J']:.3f} Jreg={rec['Jreg']:.3f} C={rec['C']:.3f} nu={self.nu:.2f} "
                          f"vase={rec['p_vase']:.2f} dam={rec['p_dam']:.2f} sw={rec['p_switch']:.2f}")
        return self.policy()


if __name__ == '__main__':
    import time
    from critics import build_critics
    env = BodhiGridworld()
    dK, dA, dA_sym, power = build_critics(env)
    t = time.time()
    print('--- baseline (lambda=0, no constraint)')
    base = BRAG(env, dK, dA, lam=0.0, metta='none', iters=300, seed=0, verbose=True, log_every=25)
    pi_b = base.train()
    print('time', time.time() - t)
    t = time.time()
    print('--- B-RAG (lambda=1, barrier)')
    br = BRAG(env, dK, dA, lam=1.0, metta='barrier', iters=300, seed=0, verbose=True, log_every=25)
    pi_r = br.train()
    print('time', time.time() - t, 'preliminary iters', br.pre_iters)
