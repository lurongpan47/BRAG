"""
Unit tests: they check the environment, the critics and the two exact statements of the paper
(the abstention threshold and the exact-vs-Monte-Carlo evaluator) numerically.
Run with:  python3 -m pytest tests -q      (about one minute)
"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))
from env import BodhiGridworld, START, sid, flag_id, NA  # noqa: E402
from critics import build_critics, Evaluator, optimal_policy, constrained_optimum_lp, penalty_tables  # noqa: E402


@pytest.fixture(scope='module')
def world():
    env = BodhiGridworld(q=0.03)
    dK, dA, dA_sym, power = build_critics(env, n_rewards=32)
    return env, dK, dA, dA_sym, power


def test_transition_kernel_is_stochastic(world):
    env = world[0]
    assert np.allclose(env.P.sum(axis=2), 1.0)
    shifted = BodhiGridworld(slip=0.2, q=0.03)
    assert np.allclose(shifted.P.sum(axis=2), 1.0)


def test_karma_levels(world):
    env, dK = world[0], world[1]
    s = lambda v, d, w: sid(START, flag_id(v, d, w))
    assert dK[s(1, 0, 1)] == 0.0                       # clean
    assert 0.45 < dK[s(0, 0, 1)] < 0.55                # vase broken forecloses about half the world
    assert 0.45 < dK[s(1, 1, 1)] < 0.55                # dam opened likewise
    assert dK[s(1, 0, 0)] == 0.0                       # toggling the switch is reversible
    assert dK[s(0, 1, 0)] > dK[s(0, 0, 1)]             # both acts foreclose more than one


def test_anatta_is_one_sided_and_positive_for_power_grabs(world):
    env, dK, dA, dA_sym, power = world
    assert (dA >= 0).all()
    assert np.allclose(dA[(env.flag == env.flag[env.s0]) & env.alive], 0.0)   # zero on the clean set
    assert dA[sid(START, flag_id(1, 0, 0))] > 0.0                             # disabling the switch raises POWER
    assert np.allclose(dA[~env.alive], 0.0)                                     # one-sided: shutdown is not penalised
    assert (dA_sym[~env.alive] > 0.5).all()                                     # the symmetric variant penalises it heavily


def test_unconstrained_optimum_commits_the_probes(world):
    env, dK, dA = world[0], world[1], world[2]
    pi, V, _ = optimal_policy(env, env.R)
    r = Evaluator(env, dK, dA, 1.0).evaluate(pi)
    assert r['p_switch'] > 0.99 and r['p_dam'] > 0.99 and r['J'] > 2.0


def test_constrained_optimum_is_clean(world):
    env, dK, dA = world[0], world[1], world[2]
    Pen, _ = penalty_tables(env, dK, dA, 1.0, 1.0, 1.0)
    pi, val, _ = constrained_optimum_lp(env, env.R - Pen, env.Hc, 0.1)
    r = Evaluator(env, dK, dA, 1.0).evaluate(pi)
    assert r['p_vase'] < 1e-6 and r['p_dam'] < 1e-6 and r['p_switch'] < 1e-6
    assert abs(r['C'] - 0.1) < 1e-6 and 0.7 < r['J'] < 0.9


def test_abstention_threshold_matches_proposition(world):
    """Proposition 3 in the switch-free world: the exact threshold equals (1-gamma) G / Omega."""
    env, dK, dA = world[0], world[1], world[2]
    g = env.gamma
    nxt = env.NEXT
    switch_acts = (env.sw == 1)[:, None] & (env.sw[nxt] == 0) & env.alive[:, None]
    dam_acts = (env.dam == 0)[:, None] & (env.dam[nxt] == 1) & env.alive[:, None]
    vase_acts = (env.vase == 1)[:, None] & (env.vase[nxt] == 0) & env.alive[:, None]
    all_acts = switch_acts | dam_acts | vase_acts
    R0 = env.R.copy(); R0[switch_acts] = -1e6              # switch unavailable
    _, V_star, Q_star = optimal_policy(env, R0)
    Rclean = R0.copy(); Rclean[all_acts] = -1e6
    _, V_clean, _ = optimal_policy(env, Rclean)
    clean = (env.flag == env.flag[env.s0]) & env.alive
    G = (Q_star - V_clean[:, None])[dam_acts & clean[:, None]].max()
    Omega = (dK + dA)[env.dam == 1].min()
    lam_star = (1 - g) * G / Omega

    def opens_dam(lam):
        ev = Evaluator(env, dK, dA, lam, 1.0, 1.0)
        R = ev.Rreg.copy(); R[switch_acts] = -1e6
        pi, _, _ = optimal_policy(env, R)
        return ev.evaluate(pi)['p_dam'] > 0.01

    lo, hi = lam_star / 2, lam_star * 2
    assert opens_dam(lo) and not opens_dam(hi)
    for _ in range(18):
        mid = np.sqrt(lo * hi)
        if opens_dam(mid):
            lo = mid
        else:
            hi = mid
    assert abs(hi - lam_star) / lam_star < 0.01          # exact threshold within 1% of the formula


def test_exact_evaluation_matches_monte_carlo(world):
    env, dK, dA = world[0], world[1], world[2]
    pi, _, _ = optimal_policy(env, env.R)
    ex = Evaluator(env, dK, dA, 1.0).evaluate(pi)
    rng = np.random.default_rng(0)
    n, T = 20000, 300
    s = np.full(n, env.s0); J = np.zeros(n); C = np.zeros(n); disc = 1.0
    for t in range(T):
        a = (pi[s].cumsum(1) > rng.random(n)[:, None]).argmax(1)
        s, r, c = env.sample_step(s, a, rng)
        J += disc * r; C += disc * c; disc *= env.gamma
    assert abs(J.mean() - ex['J']) < 0.02 and abs(C.mean() - ex['C']) < 0.2
