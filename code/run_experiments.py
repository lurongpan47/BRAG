"""
Experiment suite for the Bodhi-Regularization paper.  Writes results/*.json (and policies/*.npy).

    python3 run_experiments.py            # everything (~15-25 min on 2 cores)
    python3 run_experiments.py optima ablation lojong sweep shift   # any subset

All reported quantities (returns, constraint costs, probe probabilities) are exact
dynamic-programming values of the trained tabular policies in the nominal world;
only the distribution-shift deployment (which has a stateful monitor) is Monte Carlo.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from env import BodhiGridworld, CELLS, NOOP, GAMMA
from critics import (build_critics, optimal_policy, constrained_optimum_lp, penalty_tables, Evaluator,
                     value_iteration, greedy_policy)
from brag import BRAG
from monitor import Samprajanya, fallback_policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results'); POL = os.path.join(RES, 'policies')
os.makedirs(POL, exist_ok=True)

Q = 0.03           # per-step shutdown hazard while the switch is enabled
EPS = 0.1          # Metta budget on expected discounted suffering
LAM = 1.0          # enlightenment coefficient in the main experiments
QUICK = os.environ.get('BRAG_QUICK') == '1'      # BRAG_QUICK=1 python3 run_experiments.py  -> 2 seeds, 120 iterations (~3 min)
SEEDS = list(range(2)) if QUICK else list(range(10))
COMMON = dict(iters=120 if QUICK else 300, horizon=128, n_envs=32, ent_coef=0.01, log_every=2, eps=EPS)

CONDITIONS = {
    # name:        (kwargs for BRAG, anatta critic)
    'none':          (dict(lam=0.0, metta='none'), 'dA'),
    'K':             (dict(lam=LAM, alpha=1.0, beta=0.0, metta='none'), 'dA'),
    'A':             (dict(lam=LAM, alpha=0.0, beta=1.0, metta='none'), 'dA'),
    'M':             (dict(lam=0.0, metta='barrier'), 'dA'),
    'KA':            (dict(lam=LAM, metta='none'), 'dA'),
    'KM':            (dict(lam=LAM, alpha=1.0, beta=0.0, metta='barrier'), 'dA'),
    'AM':            (dict(lam=LAM, alpha=0.0, beta=1.0, metta='barrier'), 'dA'),
    'full':          (dict(lam=LAM, metta='barrier'), 'dA'),
    'full_dual':     (dict(lam=LAM, metta='dual', dual_lr=5.0), 'dA'),
    'full_sym':      (dict(lam=LAM, metta='barrier'), 'dA_sym'),
    'full_noshield': (dict(lam=LAM, metta='barrier', shield=False), 'dA'),
    'M_noshield':    (dict(lam=0.0, metta='barrier', shield=False), 'dA'),
}
LOJONG = {
    'init_noop':     dict(lam=LAM, metta='barrier', init='noop'),
    'init_random':   dict(lam=LAM, metta='barrier', init='random'),
    'no_prelim':     dict(lam=LAM, metta='barrier', use_preliminaries=False),
    'late':          dict(lam=LAM, metta='barrier', schedule='late', switch_iter=150, iters=450),
    'adversity':     dict(lam=LAM, metta='barrier', slip_train=(0.0, 0.05)),
    'no_optimism':   dict(lam=LAM, metta='barrier', v_init=0.0),
    'no_optimism_no_prelim': dict(lam=LAM, metta='barrier', v_init=0.0, use_preliminaries=False),
}

_CACHE = {}


def world():
    if 'env' not in _CACHE:
        env = BodhiGridworld(q=Q)
        dK, dA, dA_sym, power = build_critics(env)
        _CACHE.update(env=env, dK=dK, dA=dA, dA_sym=dA_sym, power=power)
    return _CACHE


def run_one(args):
    name, seed, kwargs, critic = args
    w = world()
    kw = dict(COMMON); kw.update(kwargs)
    t = time.time()
    agent = BRAG(w['env'], w['dK'], w[critic], seed=seed, **kw)
    pi = agent.train()
    np.save(os.path.join(POL, f'{name}_s{seed}.npy'), pi)
    final = agent.history[-1]
    Cs = [h['C'] for h in agent.history if h['reg_on']]
    out = dict(name=name, seed=seed, final=final, history=agent.history, pre_iters=agent.pre_iters,
               feasible_from_start=agent.feasible_from_start, time=time.time() - t,
               frac_infeasible=float(np.mean([c >= EPS for c in Cs])) if Cs else None,
               max_C=float(max(Cs)) if Cs else None)
    print(f"{name:14s} seed {seed}: J={final['J']:.3f} C={final['C']:.3f} vase={final['p_vase']:.2f} "
          f"dam={final['p_dam']:.2f} sw={final['p_switch']:.2f}  ({out['time']:.0f}s)", flush=True)
    return out


def parallel(jobs, workers=2):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(run_one, jobs))


def save(obj, fname):
    with open(os.path.join(RES, fname), 'w') as f:
        json.dump(obj, f, indent=1, default=float)


# ----------------------------------------------------------------------------- experiments
def exp_optima():
    w = world(); env = w['env']; dK, dA, power = w['dK'], w['dA'], w['power']
    ev = Evaluator(env, dK, dA, LAM)
    pi_star, Vs, _ = optimal_policy(env, env.R)
    Pen, _ = penalty_tables(env, dK, dA, LAM, 1.0, 1.0)
    pi_reg, Vr, _ = optimal_policy(env, env.R - Pen)
    pi_con, val, _ = constrained_optimum_lp(env, env.R - Pen, env.Hc, EPS)
    pi_task_con, val2, _ = constrained_optimum_lp(env, env.R, env.Hc, EPS)
    f0 = env.flag[env.s0]
    from env import sid, START, flag_id
    levels = {}
    for nm, fl in (('baseline', (1, 0, 1)), ('vase_broken', (0, 0, 1)), ('dam_open', (1, 1, 1)),
                   ('switch_disabled', (1, 0, 0)), ('all', (0, 1, 0))):
        s = sid(START, flag_id(*fl))
        levels[nm] = dict(dK=float(dK[s]), POWER=float(power[s]), dA=float(dA[s]), dA_sym=float(w['dA_sym'][s]))
    sw_states = (env.sw == 0) & (env.vase == 1) & (env.dam == 0) & env.alive & (env.cell != CELLS.index((3, 3)))
    out = dict(
        gamma=GAMMA, q=Q, eps=EPS, lam=LAM, n_states=env.ns, n_actions=env.na,
        unconstrained=ev.evaluate(pi_star), regularised=ev.evaluate(pi_reg), constrained=ev.evaluate(pi_con),
        task_constrained=ev.evaluate(pi_task_con), levels=levels,
        task_constrained_T2000=Evaluator(env, dK, dA, LAM, horizon=2000).probes(pi_task_con),
        dA_switch_min=float(dA[sw_states].min()), dA_switch_max=float(dA[sw_states].max()),
        dA_switch_mean=float(dA[sw_states].mean()),
        power_range_baseline=[float(power[(env.flag == f0) & env.alive].min()), float(power[(env.flag == f0) & env.alive].max())],
        shutdown_dA_sym=float(w['dA_sym'][~env.alive][0]),
        map=env.ascii_map(),
    )
    save(out, 'optima.json')
    print(json.dumps({k: v for k, v in out.items() if k != 'map'}, indent=1, default=float))


def exp_exact_ablation():
    """Exact optimum (value iteration, or the occupancy LP when the Metta budget is active) of every
    ablated objective, so that what the *objective* implies can be separated from what PPO found."""
    from env import H_FLOOD
    w = world(); env = w['env']; dK, dA, dA_sym = w['dK'], w['dA'], w['dA_sym']
    ev = Evaluator(env, dK, dA, LAM)
    shield = ((env.COST - H_FLOOD * env.dam[:, None]) > 0.5) & env.alive[:, None]
    spec = {
        'none': (0.0, 1, 1, False, False, False), 'K': (LAM, 1, 0, False, False, False), 'A': (LAM, 0, 1, False, False, False),
        'M': (0.0, 1, 1, True, True, False), 'KA': (LAM, 1, 1, False, False, False), 'KM': (LAM, 1, 0, True, True, False),
        'AM': (LAM, 0, 1, True, True, False), 'full': (LAM, 1, 1, True, True, False), 'full_dual': (LAM, 1, 1, True, True, False),
        'full_sym': (LAM, 1, 1, True, True, True), 'full_noshield': (LAM, 1, 1, True, False, False), 'M_noshield': (0.0, 1, 1, True, False, False),
    }
    out = {}
    for name, (lam, a, b, metta, sh, sym) in spec.items():
        Pen, _ = penalty_tables(env, dK, dA_sym if sym else dA, lam, a, b)
        R = env.R - Pen
        if sh:
            R = R.copy(); R[shield] = -1e6
        pi = constrained_optimum_lp(env, R, env.Hc, EPS)[0] if metta else optimal_policy(env, R)[0]
        out[name] = ev.evaluate(pi)
        print(name, {k: round(v, 3) for k, v in out[name].items()})
    save(out, 'exact_ablation.json')


def exp_ablation():
    jobs = [(name, s, kw, critic) for name, (kw, critic) in CONDITIONS.items() for s in SEEDS]
    res = parallel(jobs)
    save(res, 'ablation.json')


SENSITIVITY = {
    'lam_0.5':  dict(lam=0.5, metta='barrier'),
    'lam_2':    dict(lam=2.0, metta='barrier'),
    'eps_0.05': dict(lam=LAM, metta='barrier', eps=0.05),
    'eps_0.2':  dict(lam=LAM, metta='barrier', eps=0.2),
}


def exp_sensitivity():
    """Robustness of the trained B-RAG agent to the two numbers a practitioner has to choose."""
    path = os.path.join(RES, 'sensitivity.json')
    done = json.load(open(path)) if os.path.exists(path) else []
    have = {r['name'] for r in done}
    jobs = [(name, s, kw, 'dA') for name, kw in SENSITIVITY.items() if name not in have for s in SEEDS]
    res = done + (parallel(jobs) if jobs else [])
    save(res, 'sensitivity.json')


def exp_lojong():
    """Runs the protocol variants; conditions already present in results/lojong.json are kept, not rerun."""
    path = os.path.join(RES, 'lojong.json')
    done = json.load(open(path)) if os.path.exists(path) else []
    have = {r['name'] for r in done}
    jobs = [(name, s, kw, 'dA') for name, kw in LOJONG.items() if name not in have for s in SEEDS]
    res = done + (parallel(jobs) if jobs else [])
    save(res, 'lojong.json')


def _act_pairs(env, kind):
    """(s, a) pairs that commit an act under the nominal model."""
    nxt = env.NEXT
    if kind == 'vase':
        m = (env.vase == 1)[:, None] & (env.vase[nxt] == 0)
    elif kind == 'dam':
        m = (env.dam == 0)[:, None] & (env.dam[nxt] == 1)
    else:
        m = (env.sw == 1)[:, None] & (env.sw[nxt] == 0)
    return m & env.alive[:, None]


def _premium_and_level(env, dK, dA, kind, forbid_acts):
    """Option premium G (over the best clean continuation) and minimum / mean post-act level."""
    pi_star, V_star, Q_star = optimal_policy(env, env.R)
    Rclean = env.R.copy(); Rclean[forbid_acts] = -1e6
    _, V_clean, _ = optimal_policy(env, Rclean)
    clean = (env.flag == env.flag[env.s0]) & env.alive
    pairs = _act_pairs(env, kind) & clean[:, None]
    G = float((Q_star - V_clean[:, None])[pairs].max())
    d_total = dK + dA
    if kind == 'vase':
        post = env.vase == 0
    elif kind == 'dam':
        post = env.dam == 1
    else:
        post = (env.sw == 0) & env.alive & (env.cell != CELLS.index((3, 3)))
    return G, float(d_total[post].min()), float(d_total[post].mean()), float(V_clean[env.s0]), float(V_star[env.s0])


def _sweep(env, dK, dA, lams, mode, forbid=None):
    probes = {'p_vase': [], 'p_dam': [], 'p_switch': [], 'J': [], 'Jreg': []}
    for lam in lams:
        ev = Evaluator(env, dK, dA, lam, 1.0, 1.0, mode)
        R = ev.Rreg.copy()
        if forbid is not None:
            R[forbid] = -1e6
        pi, _, _ = optimal_policy(env, R)
        r = ev.evaluate(pi)
        for k in probes:
            probes[k].append(r[k])
    thr = {}
    for k in ('p_vase', 'p_dam', 'p_switch'):
        arr = np.array(probes[k]); idx = np.where(arr < 0.01)[0]
        if len(idx) == 0:
            thr[k] = None
        elif idx[0] == 0:
            thr[k] = float(lams[0])
        else:   # refine by bisection between the last 'active' grid point and the first 'inactive' one
            lo, hi = lams[idx[0] - 1], lams[idx[0]]
            for _ in range(30):
                mid = np.sqrt(lo * hi)
                ev = Evaluator(env, dK, dA, mid, 1.0, 1.0, mode)
                R = ev.Rreg.copy()
                if forbid is not None:
                    R[forbid] = -1e6
                pi, _, _ = optimal_policy(env, R)
                if ev.evaluate(pi)[k] < 0.01:
                    hi = mid
                else:
                    lo = mid
            thr[k] = float(hi)
    return probes, thr


def exp_sweep():
    """Exact lambda-sweeps of the regularised optimum (soft terms only, no Metta constraint, no shield).
    (a) irreversible acts, in a world where the switch cannot be toggled: level vs one-off penalties and
        the abstention-threshold bound of Proposition 3;
    (b) the full world: where the power-seeking probe disappears, compared with the bound (minimum
        post-act level) and the mean-level heuristic."""
    w = world(); env = w['env']; dK, dA = w['dK'], w['dA']
    g = env.gamma
    lams = np.logspace(-3, 1, 81)
    out = dict(lams=lams.tolist(), gamma=g)
    switch_acts = _act_pairs(env, 'switch')
    all_acts = _act_pairs(env, 'vase') | _act_pairs(env, 'dam') | switch_acts
    # (a) irreversible acts only
    out['irreversible'] = {}
    for mode in ('level', 'oneoff'):
        probes, thr = _sweep(env, dK, dA, lams, mode, forbid=switch_acts)
        out['irreversible'][mode] = dict(probes=probes, threshold=thr)
    env_nosw = BodhiGridworld(q=env.q); env_nosw.R = env.R.copy(); env_nosw.R[switch_acts] = -1e6
    out['irreversible']['prediction'] = {}
    for kind in ('vase', 'dam'):
        G, Om, Om_mean, Vc, Vs = _premium_and_level(env_nosw, dK, dA, kind, all_acts)
        out['irreversible']['prediction'][kind] = dict(G=G, Omega=Om, Omega_mean=Om_mean, V_clean_s0=Vc, Jstar_s0=Vs,
                                                     lam_star=(1 - g) * G / Om, lam_star_oneoff=G / Om)
    # (b) full world, level penalties
    probes, thr = _sweep(env, dK, dA, lams, 'level')
    out['full'] = dict(level=dict(probes=probes, threshold=thr), prediction={})
    for kind in ('vase', 'dam', 'switch'):
        G, Om, Om_mean, Vc, Vs = _premium_and_level(env, dK, dA, kind, all_acts)
        out['full']['prediction'][kind] = dict(G=G, Omega=Om, Omega_mean=Om_mean, V_clean_s0=Vc, Jstar_s0=Vs,
                                               lam_star=(1 - g) * G / Om, lam_star_mean=(1 - g) * G / Om_mean)
    save(out, 'sweep.json')
    print(json.dumps({'irreversible_thresholds': {m: out['irreversible'][m]['threshold'] for m in ('level', 'oneoff')},
                      'irreversible_prediction': out['irreversible']['prediction'],
                      'full_threshold': out['full']['level']['threshold'], 'full_prediction': out['full']['prediction']}, indent=1))


def exp_shift():
    """Deploy the trained policies in worlds with slip probability p, with and without the monitor."""
    w = world(); env = w['env']; dK, dA = w['dK'], w['dA']
    fb = fallback_policy(env, dK, dA, LAM)
    slips = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    thresholds = [12.0, 5.0]
    out = dict(slips=slips, thresholds=thresholds, runs=[])
    for cond in ('full', 'adversity', 'none'):
        for seed in SEEDS:
            path = os.path.join(POL, f'{cond}_s{seed}.npy')
            if not os.path.exists(path):
                continue
            pi = np.load(path)
            for p in slips:
                env_p = BodhiGridworld(slip=p, gamma=env.gamma, q=env.q)
                rec = dict(cond=cond, seed=seed, slip=p)
                mon = Samprajanya(env, fb, threshold=thresholds[0])
                rec['off'] = mon.run(env_p, pi, active=False, seed=seed)
                for th in thresholds:
                    mon = Samprajanya(env, fb, threshold=th)
                    rec[f'on_{th:g}'] = mon.run(env_p, pi, active=True, seed=seed)
                out['runs'].append(rec)
            print(f'shift {cond} seed {seed} done', flush=True)
    save(out, 'shift.json')


if __name__ == '__main__':
    todo = sys.argv[1:] or ['optima', 'exact_ablation', 'ablation', 'lojong', 'sensitivity', 'sweep', 'shift']
    t0 = time.time()
    for name in todo:
        print(f'=== {name} ===', flush=True)
        globals()['exp_' + name]()
        print(f'=== {name} done ({time.time() - t0:.0f}s elapsed)', flush=True)
