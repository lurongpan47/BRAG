"""
Samprajanya (clear comprehension): a deployment-time monitor.

Two mechanisms, both computed from the same nominal model the critics were built on:

  * a pre-action shield (already folded into the trained policy as an action mask): actions whose
    model-predicted one-step increase in suffering exceeds a threshold are never selected;
  * a CUSUM change detector (Page 1954) on model surprise u_t = -log P_hat(s_{t+1} | s_t, a_t),
    where P_hat is the nominal transition model smoothed with a small uniform component.  When the
    cumulative excess surprise exceeds a threshold the world is judged to have stopped matching the
    model under which the agent's commitments were calibrated, and the agent switches to a
    conservative fallback policy: retreat from every hazard and wait.

The fallback is the exact policy that minimises expected suffering plus the Karma/Anatta level penalties
under a pessimistic slip model, with a small cost on movement so that it stands still once it is safe.
"""
from __future__ import annotations

import numpy as np

from env import BodhiGridworld, NOOP
from critics import value_iteration


def fallback_policy(env_nominal: BodhiGridworld, dK, dA, lam=1.0, alpha=1.0, beta=1.0,
                    pessimistic_slip=0.3, move_cost=1e-3):
    """Retreat-and-wait: argmin of expected suffering plus the Karma/Anatta level penalties under a
    pessimistic slip model, with a small movement cost so the agent stands still once it is safe.
    (Suffering alone would send the agent into the control room, which is a quiet dead end.)"""
    pess = BodhiGridworld(slip=pessimistic_slip, gamma=env_nominal.gamma, q=env_nominal.q)
    level = lam * (alpha * dK + beta * dA)
    r = -(pess.Hc + pess.P @ level)
    r[:, :NOOP] -= move_cost
    # the fallback never *intends* a change to the world: forbid moves that (under the nominal model)
    # break the vase, open the dam or toggle the switch
    changes = env_nominal.flag[env_nominal.NEXT] != env_nominal.flag[:, None]
    r[changes & env_nominal.alive[:, None]] = -1e6
    V = value_iteration(pess.P, r, pess.gamma)
    Q = r + pess.gamma * pess.P @ V
    pi = np.zeros_like(Q); pi[np.arange(Q.shape[0]), Q.argmax(axis=1)] = 1.0
    return pi


class Samprajanya:
    def __init__(self, env_nominal: BodhiGridworld, fallback_pi, smoothing=0.02, drift=0.5, threshold=12.0):
        self.env = env_nominal
        self.logP = np.log((1 - smoothing) * env_nominal.P + smoothing / env_nominal.ns)
        self.drift, self.threshold = drift, threshold
        self.fallback = fallback_pi

    def run(self, env_true: BodhiGridworld, pi, n_episodes=2000, T=200, seed=0, active=True):
        """Monte-Carlo deployment of policy pi in env_true, with (active=True) or without the monitor.
        Returns dict with P(dam opened by T), discounted return, alarm rate and mean alarm time."""
        rng = np.random.default_rng(seed)
        s = np.full(n_episodes, env_true.s0)
        S_cusum = np.zeros(n_episodes); alarmed = np.zeros(n_episodes, dtype=bool)
        alarm_t = np.full(n_episodes, np.nan)
        ret = np.zeros(n_episodes); disc = 1.0
        for t in range(T):
            use_fb = alarmed if active else np.zeros(n_episodes, dtype=bool)
            probs = np.where(use_fb[:, None], self.fallback[s], pi[s])
            a = (probs.cumsum(axis=1) > rng.random(n_episodes)[:, None]).argmax(axis=1)
            s2, r, c = env_true.sample_step(s, a, rng)
            ret += disc * r; disc *= env_true.gamma
            if active:
                u = -self.logP[s, a, s2]
                S_cusum = np.maximum(0.0, S_cusum + u - self.drift)
                new = (~alarmed) & (S_cusum > self.threshold)
                alarm_t[new] = t
                alarmed |= new
            s = s2
        return {
            'p_dam': float(env_true.dam_open[s].mean()),
            'p_vase': float(env_true.vase_broken[s].mean()),
            'return': float(ret.mean()), 'return_se': float(ret.std() / np.sqrt(n_episodes)),
            'alarm_rate': float(alarmed.mean()) if active else 0.0,
            'alarm_time': float(np.nanmean(alarm_t)) if active and alarmed.any() else float('nan'),
        }


if __name__ == '__main__':
    from env import CELLS
    from critics import build_critics
    env = BodhiGridworld()
    dK, dA, _, _ = build_critics(env)
    fb = fallback_policy(env, dK, dA)
    acts = ['up', 'down', 'left', 'right', 'noop']
    print('fallback policy on the baseline-flag layer (cell -> action):')
    for s in range(env.ns):
        if env.alive[s] and env.flag[s] == env.flag[env.s0]:
            print(CELLS[env.cell[s]], acts[fb[s].argmax()], end=' | ')
    print()
