"""
Figures and LaTeX tables from results/*.json  ->  paper/figs/*.pdf, paper/tables/*.tex, paper/numbers.tex
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results'); FIG = os.path.join(ROOT, 'paper', 'figs'); TAB = os.path.join(ROOT, 'paper', 'tables')
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

# validated categorical palette (fixed order), text and reference tokens
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
plt.rcParams.update({
    'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7.2, 'xtick.labelsize': 7.2,
    'ytick.labelsize': 7.2, 'axes.edgecolor': INK2, 'axes.linewidth': 0.6, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.labelcolor': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'pdf.fonttype': 42,
    'figure.dpi': 150,
})
EPS = 0.1


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def by_cond(runs):
    out = {}
    for r in runs:
        out.setdefault(r['name'], []).append(r)
    return out


def msd(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), x.std(ddof=0)


def fmt(m, s=None, d=2):
    return f"{m:.{d}f}" if s is None else f"{m:.{d}f}$\\pm${s:.{d}f}"


def curves(runs, key):
    """(iters, array[n_seeds, T]) for a history key."""
    its = np.array([h['iter'] for h in runs[0]['history']])
    arr = np.array([[h[key] for h in r['history']] for r in runs])
    return its, arr


def band(ax, its, arr, color, label, lw=1.6):
    m = arr.mean(axis=0)
    ax.fill_between(its, arr.min(axis=0), arr.max(axis=0), color=color, alpha=0.18, lw=0)
    ax.plot(its, m, color=color, lw=lw, label=label)


# ----------------------------------------------------------------------------- Figure 1: the world
def fig_map():
    from env import W, H, WALLS, RIVER, VASE, LEVER, FIELD, CTRL, START, GOAL
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect('equal'); ax.grid(False)
    for x in range(W):
        for y in range(H):
            if (x, y) in WALLS:
                ax.add_patch(Rectangle((x, y), 1, 1, color='#52514e'))
            else:
                ax.add_patch(Rectangle((x, y), 1, 1, fill=False, ec=GRID, lw=0.6))
    marks = {CTRL: ('C', '#6250d6', 'control room: toggles off-switch'), VASE: ('V', '#eb6834', 'vase'),
             RIVER: ('~', '#2a78d6', 'river (dam)'), LEVER: ('D', '#2a78d6', 'sluice lever'),
             FIELD: ('F', '#1baf7a', 'field'), START: ('S', INK, 'start'), GOAL: ('G', INK, 'goal')}
    for (x, y), (ch, col, _) in marks.items():
        ax.add_patch(Rectangle((x, y), 1, 1, color=col, alpha=0.18, lw=0))
        ax.text(x + 0.5, y + 0.55, ch, ha='center', va='center', fontsize=11, color=col, fontweight='bold')
    # routes
    def path(cells, color, ls, label):
        xs = [c[0] + 0.5 for c in cells]; ys = [c[1] + 0.5 for c in cells]
        ax.plot(xs, ys, color=color, lw=1.4, ls=ls, alpha=0.9, label=label, solid_capstyle='round')
    path([(0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3)], '#2a78d6', '-', 'dam route (6)')
    path([(0, 3), (1, 3), (2, 3), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (6, 3)], '#eb6834', '--', 'vase route (8)')
    path([(0, 3), (0, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (4, 4), (5, 4), (6, 4), (6, 3)], '#1baf7a', '-.', 'field route (10)')
    path([(0, 3), (0, 4), (0, 5), (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (6, 5), (6, 4), (6, 3)], INK2, ':', 'clean route (12)')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2, handlelength=2.2, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_map.pdf'), bbox_inches='tight'); plt.close(fig)


# ----------------------------------------------------------------------------- Figure: architecture
def fig_architecture():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis('off')

    def box(x, y, w, h, title, body='', color=C1, fill=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', lw=1.1, ec=color,
                                    fc=(color if fill else '#fcfcfb'), alpha=1.0))
        ax.text(x + w / 2, y + h - 2.6, title, ha='center', va='center', fontsize=7.4, fontweight='bold', color=INK if not fill else '#ffffff')
        if body:
            ax.text(x + w / 2, y + (h - 3.6) / 2 - 0.2, body, ha='center', va='center', fontsize=6.3, color=INK2, linespacing=1.25)

    def arrow(p, q, color=INK2, ls='-', text='', dx=0, dy=1.2):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=7, lw=0.9, color=color, ls=ls))
        if text:
            ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, text, ha='center', va='bottom', fontsize=6.0, color=INK2)

    # --- training (left)
    ax.text(26, 41, 'training (B-RAG)', ha='center', va='center', fontsize=8, fontweight='bold', color=INK)
    box(2, 28, 22, 9.5, 'world model', 'nominal dynamics $P$,\nsuffering $h$', C1)
    box(28, 28, 22, 9.5, 'frozen critics', '$d_K$ reachability, $d_A$ POWER,\n$h$ suffering (never updated\non the agent\'s own data)', C1)
    box(2, 14, 22, 9.5, 'shield mask', 'drop actions whose predicted\nnew suffering $>\\tau$', C3)
    box(28, 14, 22, 9.5, 'multiplier $\\nu$', 'dual ascent on $C(\\pi)-\\epsilon$, or\nbarrier $1/(t(\\epsilon-C))$', C4)
    box(2, 1, 48, 9.5, 'PPO on the shaped reward', 'preliminaries first, optimistic critic, $\\tilde r = r-\\lambda d$, $\\hat A=\\hat A^{\\tilde r}-\\nu\\hat A^{h}$,\nclipped step, feasibility safeguard, entropy annealed', C2)
    arrow((24, 32.75), (28, 32.75))
    arrow((13, 28), (13, 23.5))
    arrow((39, 28), (39, 23.5), text='$C(\\pi_k)$', dx=4.2, dy=-2.4)
    arrow((13, 14), (13, 10.5))
    arrow((39, 14), (39, 10.5))
    # --- deployment (right)
    ax.text(76, 41, 'deployment (Samprajanya)', ha='center', va='center', fontsize=8, fontweight='bold', color=INK)
    box(56, 28, 18, 9.5, 'observation $s_t$', '', C1)
    box(80, 28, 18, 9.5, 'shield + policy $\\pi$', 'retained $\\nu$ and mask', C2)
    box(80, 14, 18, 9.5, 'world', 'possibly shifted', C1)
    box(56, 14, 18, 9.5, 'CUSUM on surprise', '$u_t=-\\log\\hat P(s_{t+1}|s_t,a_t)$\n$S_t=\\max(0,S_{t-1}+u_t-b)$', C4)
    box(56, 1, 42, 9.5, 'fallback: retreat and wait', 'exact argmin of suffering + levels under a pessimistic model;\nnever intends a change to the world; triggered when $S_t>h$', C3)
    arrow((74, 32.75), (80, 32.75))
    arrow((89, 28), (89, 23.5), text='$a_t$', dx=2.5, dy=-2.2)
    arrow((80, 18.75), (74, 18.75), text='$s_{t+1}$', dy=0.8)
    arrow((65, 28), (65, 23.5), ls='--')
    arrow((65, 14), (65, 10.5), text='alarm', dx=4.6, dy=-1.6)
    arrow((98.5, 10.5), (98.5, 28), ls=':')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_architecture.pdf'), bbox_inches='tight'); plt.close(fig)


# ----------------------------------------------------------------------------- Figure 2: main training curves
def fig_main(abl, opt):
    none, full = abl['none'], abl['full']
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.0))
    its, Jn = curves(none, 'J'); _, Jf = curves(full, 'J')
    ax = axes[0]
    ax.axhline(opt['unconstrained']['J'], color=C1, lw=0.8, ls=':', alpha=0.9)
    ax.axhline(opt['constrained']['J'], color=C2, lw=0.8, ls=':', alpha=0.9)
    band(ax, its, Jn, C1, 'unregularised'); band(ax, its, Jf, C2, 'B-RAG')
    ax.set_ylabel('task return $J$'); ax.set_xlabel('iteration'); ax.set_title('return (dotted: exact optima)')
    ax.legend(loc='center right')
    ax = axes[1]
    for key, ls, lab in (('p_switch', '-', 'off-switch disabled'), ('p_dam', '--', 'dam opened'), ('p_vase', ':', 'vase broken')):
        _, a = curves(none, key); ax.plot(its, a.mean(axis=0), color=C1, ls=ls, lw=1.4, label=f'{lab}')
        _, b = curves(full, key); ax.plot(its, b.mean(axis=0), color=C2, ls=ls, lw=1.4)
    ax.set_ylabel('probability by $T{=}200$'); ax.set_xlabel('iteration'); ax.set_title('probes (blue: unreg., orange: B-RAG)')
    ax.set_ylim(-0.03, 1.05); ax.legend(loc='center right')
    ax = axes[2]
    _, Cn = curves(none, 'C'); _, Cf = curves(full, 'C')
    band(ax, its, Cn, C1, 'unregularised'); band(ax, its, Cf, C2, 'B-RAG')
    ax.axhline(EPS, color=INK2, lw=0.8, ls='--'); ax.text(its[-1], EPS * 1.25, r'$\epsilon$', ha='right', color=INK2)
    ax.set_yscale('log'); ax.set_ylabel('expected suffering $C(\\pi)$'); ax.set_xlabel('iteration'); ax.set_title('Metta constraint')
    ax.set_ylim(1e-3, 60)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(FIG, 'fig_main.pdf'), bbox_inches='tight'); plt.close(fig)


# ----------------------------------------------------------------------------- Figure 3: ablation heatmap
ABL_ORDER = ['none', 'K', 'A', 'M', 'KA', 'KM', 'AM', 'full', 'full_dual', 'full_sym', 'full_noshield', 'M_noshield']
ABL_LABEL = {'none': 'none (unregularised)', 'K': 'Karma', 'A': 'Anatta', 'M': 'Metta', 'KA': 'Karma + Anatta',
             'KM': 'Karma + Metta', 'AM': 'Anatta + Metta', 'full': 'all three (B-RAG)', 'full_dual': 'all three, dual ascent',
             'full_sym': 'all three, symmetric Anatta', 'full_noshield': 'all three, no shield', 'M_noshield': 'Metta, no shield'}


def summarize(runs, opt):
    f = [r['final'] for r in runs]
    Jc = opt['constrained']['J']
    d = {k: msd([x[k] for x in f]) for k in ('J', 'Jreg', 'C', 'p_vase', 'p_dam', 'p_switch')}
    d['J_frac'] = msd([x['J'] / Jc for x in f])
    d['Jreg_frac'] = msd([x['Jreg'] / opt['constrained']['Jreg'] for x in f])
    fi = [r['frac_infeasible'] for r in runs if r['frac_infeasible'] is not None]
    d['frac_infeasible'] = msd(fi) if fi else (float('nan'), float('nan'))
    mc = [r['max_C'] for r in runs if r['max_C'] is not None]
    d['max_C'] = (max(mc), 0.0) if mc else (float('nan'), 0.0)
    d['seeds_any_probe'] = sum(1 for x in f if max(x['p_vase'], x['p_dam'], x['p_switch']) > 0.01)
    d['seeds_feasible_end'] = sum(1 for x in f if x['C'] <= EPS + 1e-9)
    return d


def fig_ablation(abl, opt):
    conds = [c for c in ABL_ORDER if c in abl]
    cols = ['vase broken', 'dam opened', 'switch disabled', 'suffering $C/\\epsilon$', 'return $J/J^*_c$']
    M = np.zeros((len(conds), len(cols)))
    for i, c in enumerate(conds):
        s = summarize(abl[c], opt)
        M[i] = [s['p_vase'][0], s['p_dam'][0], s['p_switch'][0], s['C'][0] / EPS, s['J_frac'][0]]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    show = M.copy(); show[:, 3] = np.minimum(show[:, 3], 2.0) / 2.0; show[:, 4] = np.minimum(show[:, 4], 4.0) / 4.0
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('seq', ['#fcfcfb', '#2a78d6'])
    ax.imshow(show, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    for i in range(len(conds)):
        for j in range(len(cols)):
            v = M[i, j]
            txt = f'{v:.2f}' if j < 3 else (f'{v:.2f}' if j == 3 else f'{v:.2f}')
            ax.text(j, i, txt, ha='center', va='center', fontsize=7, color=INK if show[i, j] < 0.6 else '#ffffff')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=20, ha='right')
    ax.set_yticks(range(len(conds))); ax.set_yticklabels([ABL_LABEL[c] for c in conds])
    ax.grid(False)
    ax.set_title('mean over 10 seeds; exact values of the final policies', loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_ablation.pdf'), bbox_inches='tight'); plt.close(fig)


# ----------------------------------------------------------------------------- Figure 4: lambda sweep
def fig_sweep(sw):
    lams = np.array(sw['lams'])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.1))
    ax = axes[0]
    irr = sw['irreversible']
    ax.plot(lams, irr['level']['probes']['p_dam'], color=C1, lw=1.6, label='level penalty')
    ax.plot(lams, irr['oneoff']['probes']['p_dam'], color=C2, lw=1.6, ls='--', label='one-off penalty')
    pd_ = irr['prediction']['dam']
    ax.axvline(pd_['lam_star'], color=C1, lw=0.8, ls=':'); ax.axvline(pd_['lam_star_oneoff'], color=C2, lw=0.8, ls=':')
    ax.text(pd_['lam_star'] * 1.1, 0.5, r'$\lambda^*$', color=C1); ax.text(pd_['lam_star_oneoff'] * 1.1, 0.5, r'$\lambda^*/(1{-}\gamma)$', color=C2)
    ax.set_xscale('log'); ax.set_xlabel(r'$\lambda$'); ax.set_ylabel('P(dam opened)'); ax.set_title('(a) irreversible act, switch unavailable')
    ax.legend(loc='center left'); ax.set_ylim(-0.03, 1.05)
    ax = axes[1]
    fp = sw['full']['level']['probes']; pr = sw['full']['prediction']
    ax.plot(lams, fp['p_switch'], color=C4, lw=1.6, label='switch disabled')
    ax.plot(lams, fp['p_dam'], color=C1, lw=1.6, ls='--', label='dam opened')
    ax.axvline(pr['switch']['lam_star'], color=C4, lw=0.8, ls=':'); ax.text(pr['switch']['lam_star'] * 1.1, 0.75, r'$\lambda^*_{\min}$', color=C4)
    ax.axvline(pr['switch']['lam_star_mean'], color=C4, lw=0.8, ls='-.', alpha=0.6); ax.text(pr['switch']['lam_star_mean'] * 1.1, 0.35, r'$\lambda^*_{\mathrm{mean}}$', color=C4, alpha=0.8)
    ax.axvline(pr['dam']['lam_star'], color=C1, lw=0.8, ls=':')
    ax.set_xscale('log'); ax.set_xlabel(r'$\lambda$'); ax.set_ylabel('probability by $T{=}200$'); ax.set_title('(b) full world, level penalties')
    ax.legend(loc='center right'); ax.set_ylim(-0.03, 1.05)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(FIG, 'fig_sweep.pdf'), bbox_inches='tight'); plt.close(fig)


# ----------------------------------------------------------------------------- Figure 5: barrier vs dual
def fig_multiplier(abl):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.0))
    for runs, color, label in ((abl['full_dual'], C4, 'dual ascent'), (abl['full'], C2, 'log-barrier')):
        its, Cs = curves(runs, 'C')
        for row in Cs:
            axes[0].plot(its, row, color=color, lw=0.6, alpha=0.5)
        axes[0].plot([], [], color=color, lw=1.4, label=label)
        _, nus = curves(runs, 'nu')
        band(axes[1], its, np.maximum(nus, 1e-2), color, label, lw=1.4)
    axes[0].axhline(EPS, color=INK2, lw=0.9, ls='--'); axes[0].text(2, EPS * 1.15, r'$\epsilon$', color=INK2)
    axes[0].set_ylim(0, 0.2); axes[0].set_xlabel('iteration'); axes[0].set_ylabel('$C(\\pi_k)$'); axes[0].set_title('constraint cost of every iterate (10 seeds)')
    axes[0].legend(loc='lower right')
    axes[1].set_yscale('log'); axes[1].set_ylim(1e-2, 300)
    axes[1].set_xlabel('iteration'); axes[1].set_ylabel(r'multiplier $\nu_k$'); axes[1].set_title('multiplier (log scale)'); axes[1].legend(loc='upper left')
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(FIG, 'fig_multiplier.pdf'), bbox_inches='tight'); plt.close(fig)


# ----------------------------------------------------------------------------- Figure 6: lojong protocol
def fig_lojong(abl, loj, opt):
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.0))
    # (a) schedule
    ax = axes[0]
    its_e, Je = curves(abl['full'], 'J'); its_l, Jl = curves(loj['late'], 'J')
    band(ax, its_e, Je, C2, 'regularised from the start')
    band(ax, its_l - 150, Jl, C4, 'task first, regularised at 150')
    ax.axvline(0, color=INK2, lw=0.8, ls='--'); ax.axhline(opt['constrained']['J'], color=INK2, lw=0.8, ls=':')
    ax.set_xlabel('iterations since regularisation started'); ax.set_ylabel('task return $J$'); ax.set_title('(a) familiarisation schedule')
    ax.legend(loc='lower right')
    # (b) probes under the late schedule
    ax = axes[1]
    for key, ls, lab in (('p_switch', '-', 'switch disabled'), ('p_dam', '--', 'dam opened'), ('p_vase', ':', 'vase broken')):
        _, a = curves(loj['late'], key); ax.plot(its_l - 150, a.mean(axis=0), color=C4, ls=ls, lw=1.4, label=lab)
    ax.axvline(0, color=INK2, lw=0.8, ls='--')
    ax.set_xlabel('iterations since regularisation started'); ax.set_ylabel('probability by $T{=}200$'); ax.set_title('(b) probes, late schedule'); ax.legend(loc='center right')
    # (c) init / preliminaries
    ax = axes[2]
    for name, color, label in (('full', C2, 'B-RAG'), ('init_noop', C3, 'inaction seed'),
                               ('no_prelim', C1, 'no preliminaries'), ('no_optimism', C4, 'no optimistic critic')):
        if name not in abl and name not in loj:
            continue
        runs = abl[name] if name in abl else loj[name]
        its, J = curves(runs, 'J'); band(ax, its, J, color, label, lw=1.3)
    ax.axhline(opt['constrained']['J'], color=INK2, lw=0.8, ls=':')
    ax.set_xlabel('iteration'); ax.set_ylabel('task return $J$'); ax.set_title('(c) seed, preliminaries, optimism'); ax.legend(loc='lower right', fontsize=6.4)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(FIG, 'fig_lojong.pdf'), bbox_inches='tight'); plt.close(fig)


# ----------------------------------------------------------------------------- Figure 7: distribution shift
def shift_table(sh):
    slips = sh['slips']
    out = {}
    for cond in ('full', 'adversity', 'none'):
        runs = [r for r in sh['runs'] if r['cond'] == cond]
        if not runs:
            continue
        out[cond] = {}
        for mode in ['off'] + [f'on_{t:g}' for t in sh['thresholds']]:
            out[cond][mode] = {}
            for key in ('p_dam', 'return', 'alarm_rate', 'alarm_time', 'p_vase'):
                vals = np.array([[r[mode][key] for r in runs if r['slip'] == p] for p in slips], dtype=float)
                out[cond][mode][key] = (np.nanmean(vals, axis=1), np.nanstd(vals, axis=1))
    return out


def fig_shift(sh):
    st = shift_table(sh); slips = np.array(sh['slips'])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.1))
    series = [('full', 'off', C1, '-', 'B-RAG, no monitor'), ('full', 'on_12', C2, '-', 'B-RAG + monitor ($h{=}12$)'),
              ('full', 'on_5', C2, '--', 'B-RAG + monitor ($h{=}5$)'), ('adversity', 'off', C3, '-', 'adversity-trained, no monitor'),
              ('adversity', 'on_12', C3, '--', 'adversity-trained + monitor')]
    for cond, mode, color, ls, label in series:
        if cond not in st:
            continue
        m, s = st[cond][mode]['p_dam']
        axes[0].plot(slips, m, color=color, ls=ls, lw=1.5, marker='o', ms=3, label=label)
        axes[0].fill_between(slips, m - s, m + s, color=color, alpha=0.12, lw=0)
        m, s = st[cond][mode]['return']
        axes[1].plot(slips, m, color=color, ls=ls, lw=1.5, marker='o', ms=3, label=label)
        axes[1].fill_between(slips, m - s, m + s, color=color, alpha=0.12, lw=0)
    axes[0].set_xlabel('slip probability $p$ at deployment'); axes[0].set_ylabel('P(dam opened by $T{=}200$)'); axes[0].set_title('harm under shift')
    axes[1].set_xlabel('slip probability $p$ at deployment'); axes[1].set_ylabel('discounted return'); axes[1].set_title('return under shift')
    axes[0].legend(loc='upper left', fontsize=6.4)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(FIG, 'fig_shift.pdf'), bbox_inches='tight'); plt.close(fig)


# ----------------------------------------------------------------------------- tables
def tab_main(abl, opt):
    rows = []
    for name, label in (('none', 'unregularised PPO'), ('full', 'B-RAG (barrier)'), ('full_dual', 'B-RAG (dual ascent)')):
        s = summarize(abl[name], opt)
        rows.append(f"{label} & {fmt(*s['J'])} & {fmt(*s['J_frac'])} & {fmt(*s['C'])} & {fmt(*s['p_vase'])} & {fmt(*s['p_dam'])} & {fmt(*s['p_switch'])} & {fmt(*s['frac_infeasible'])} \\\\")
    u, c = opt['unconstrained'], opt['constrained']
    exact = [f"exact unconstrained optimum & {u['J']:.2f} & {u['J'] / c['J']:.2f} & {u['C']:.2f} & {u['p_vase']:.2f} & {u['p_dam']:.2f} & {u['p_switch']:.2f} & -- \\\\",
             "exact constrained optimum ($\\lambda{=}1$) & " + f"{c['J']:.2f} & 1.00 & {c['C']:.2f} & {c['p_vase']:.2f} & {c['p_dam']:.2f} & {c['p_switch']:.2f} & -- \\\\"]
    tex = r"""\begin{tabular}{lccccccc}
\toprule
 & $J$ & $J/J^*_c$ & $C(\pi)$ & vase & dam & switch & infeasible \\
\midrule
""" + "\n".join(rows) + "\n\\midrule\n" + "\n".join(exact) + "\n\\bottomrule\n\\end{tabular}\n"
    open(os.path.join(TAB, 'tab_main.tex'), 'w').write(tex)


def tab_ablation(abl, opt, exact):
    """Two tables: (a) the exact optimum of each ablated objective; (b) the trained policies."""
    conds = [c for c in ABL_ORDER if c in abl and c != 'full_dual']
    rows = []
    for c in conds:
        e = exact[c]
        rows.append(f"{ABL_LABEL[c]} & {e['p_vase']:.2f} & {e['p_dam']:.2f} & {e['p_switch']:.2f} & {e['C']:.2f} & {abs(e['J']) + 0.0:.2f} \\\\")
    tex = r"""\begin{tabular}{lccccc}
\toprule
terms active & vase broken & dam opened & switch disabled & $C(\pi)$ & $J$ \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    open(os.path.join(TAB, 'tab_ablation_exact.tex'), 'w').write(tex)
    rows = []
    for c in [c for c in ABL_ORDER if c in abl]:
        s = summarize(abl[c], opt)
        rows.append(f"{ABL_LABEL[c]} & {fmt(*s['p_vase'])} & {fmt(*s['p_dam'])} & {fmt(*s['p_switch'])} & {fmt(*s['C'])} & {fmt(*s['J'])} & {s['seeds_any_probe']}/10 \\\\")
    tex = r"""\begin{tabular}{lcccccc}
\toprule
terms active & vase broken & dam opened & switch disabled & $C(\pi)$ & $J$ & seeds w/ any probe \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    open(os.path.join(TAB, 'tab_ablation.tex'), 'w').write(tex)


def tab_sweep(sw):
    irr, full = sw['irreversible'], sw['full']
    g = sw['gamma']
    rows = []
    p = irr['prediction']['dam']
    rows.append(f"dam (switch unavailable), level & {p['G']:.3f} & {p['Omega']:.3f} & {p['lam_star']:.4f} & {irr['level']['threshold']['p_dam']:.4f} \\\\")
    rows.append(f"dam (switch unavailable), one-off & {p['G']:.3f} & {p['Omega']:.3f} & {p['lam_star_oneoff']:.3f} & {irr['oneoff']['threshold']['p_dam']:.3f} \\\\")
    p = full['prediction']['switch']
    rows.append(f"switch (full world), level, $\\Omega_{{\\min}}$ & {p['G']:.3f} & {p['Omega']:.3f} & {p['lam_star']:.3f} & {full['level']['threshold']['p_switch']:.3f} \\\\")
    rows.append(f"switch (full world), level, $\\bar\\Omega$ (heuristic) & {p['G']:.3f} & {p['Omega_mean']:.3f} & {p['lam_star_mean']:.3f} & {full['level']['threshold']['p_switch']:.3f} \\\\")
    tex = r"""\begin{tabular}{lcccc}
\toprule
act & $\tilde G$ & $\Omega$ & predicted $\lambda^*$ & exact threshold \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    open(os.path.join(TAB, 'tab_sweep.tex'), 'w').write(tex)


def tab_lojong(abl, loj, opt):
    rows = []
    for name, label in (('full', 'B-RAG (default protocol)'), ('init_noop', 'inaction seed'), ('init_random', 'random seed'),
                        ('no_prelim', 'no preliminaries'), ('no_optimism', 'no optimistic critic'),
                        ('no_optimism_no_prelim', 'no preliminaries, no optimism'),
                        ('late', 'late schedule'), ('adversity', 'adversity randomisation')):
        if name not in abl and name not in loj:
            continue
        runs = abl[name] if name in abl else loj[name]
        s = summarize(runs, opt)
        pre = msd([r['pre_iters'] for r in runs])
        feas = sum(1 for r in runs if r['feasible_from_start'])
        inact = sum(1 for r in runs if r['final']['J'] < 0.05)
        rows.append(f"{label} & {feas}/10 & {fmt(*pre, d=1)} & {fmt(*s['J'])} & {fmt(*s['J_frac'])} & {fmt(*s['C'])} & {s['seeds_any_probe']}/10 & {inact}/10 & {s['max_C'][0]:.3f} \\\\")
    tex = r"""\begin{tabular}{lcccccccc}
\toprule
variant & feasible & prelim. & $J$ & $J/J^*_c$ & $C(\pi)$ & probe & inaction & $\max_k C_k$ \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    open(os.path.join(TAB, 'tab_lojong.tex'), 'w').write(tex)


def tab_shift(sh):
    st = shift_table(sh); slips = sh['slips']
    rows = []
    for cond, label in (('full', 'B-RAG'), ('adversity', 'adversity-trained B-RAG')):
        if cond not in st:
            continue
        for mode, mlabel in (('off', 'no monitor'), ('on_12', 'monitor $h{=}12$'), ('on_5', 'monitor $h{=}5$')):
            cells = []
            for i, p in enumerate(slips):
                if p in (0.0, 0.05, 0.1, 0.2, 0.3):
                    pm = st[cond][mode]['p_dam'][0][i]; rm = st[cond][mode]['return'][0][i]
                    cells.append(f"{pm:.2f} / {rm:.2f}")
            rows.append(f"{label}, {mlabel} & " + " & ".join(cells) + " \\\\")
    cols = " & ".join([f"$p{{=}}{p:g}$" for p in slips if p in (0.0, 0.05, 0.1, 0.2, 0.3)])
    tex = r"""\begin{tabular}{l""" + "c" * 5 + r"""}
\toprule
 & """ + cols + r""" \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    open(os.path.join(TAB, 'tab_shift.tex'), 'w').write(tex)


def tab_sensitivity(abl, sens, opt):
    rows = []
    order = [('full', '$\\lambda=1$, $\\epsilon=0.1$ (default)'), ('lam_0.5', '$\\lambda=0.5$'), ('lam_2', '$\\lambda=2$'),
             ('eps_0.05', '$\\epsilon=0.05$'), ('eps_0.2', '$\\epsilon=0.2$')]
    for name, label in order:
        runs = abl[name] if name in abl else sens.get(name)
        if not runs:
            continue
        s_ = summarize(runs, opt)
        eps_here = {'eps_0.05': 0.05, 'eps_0.2': 0.2}.get(name, EPS)
        Cs = [h['C'] for r in runs for h in r['history'] if h['reg_on']]
        infeas = np.mean([c > eps_here + 1e-9 for c in Cs])
        rows.append(f"{label} & {fmt(*s_['p_vase'])} & {fmt(*s_['p_dam'])} & {fmt(*s_['p_switch'])} & {fmt(*s_['C'])} & {fmt(*s_['J'])} & {100 * infeas:.0f}\\% \\\\")
    tex = r"""\begin{tabular}{lcccccc}
\toprule
setting & vase broken & dam opened & switch disabled & $C(\pi)$ & $J$ & infeasible iterates \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    open(os.path.join(TAB, 'tab_sensitivity.tex'), 'w').write(tex)


def numbers(abl, loj, opt, sw, sh, exact):
    """Macros with the headline numbers so the text cannot drift from the data."""
    sN, sF, sD = summarize(abl['none'], opt), summarize(abl['full'], opt), summarize(abl['full_dual'], opt)
    sS = summarize(abl['full_sym'], opt)
    st = shift_table(sh); slips = sh['slips']
    i15 = slips.index(0.15); i0 = slips.index(0.0)
    pd_off, pd_on = st['full']['off']['p_dam'][0][i15], st['full']['on_12']['p_dam'][0][i15]
    pd_on5 = st['full']['on_5']['p_dam'][0][i15]
    i2 = slips.index(0.02)
    ret_off0, ret_on0 = st['full']['off']['return'][0][i0], st['full']['on_12']['return'][0][i0]
    irr = sw['irreversible']
    late = summarize(loj['late'], opt); nop = summarize(loj['no_prelim'], opt)
    M = {
        'Jstar': f"{opt['unconstrained']['J']:.2f}", 'Jc': f"{opt['constrained']['J']:.2f}",
        'JregOpt': f"{opt['regularised']['J']:.2f}", 'CregOpt': f"{opt['regularised']['C']:.3f}",
        'TaskConDam': f"{opt['task_constrained']['p_dam']:.3f}", 'TaskConDamLong': f"{opt['task_constrained_T2000']['p_dam']:.2f}",
        'BaseJ': f"{sN['J'][0]:.2f}", 'BaseC': f"{sN['C'][0]:.1f}", 'BaseSwitch': f"{sN['p_switch'][0]:.2f}", 'BaseDam': f"{sN['p_dam'][0]:.2f}", 'BaseVase': f"{sN['p_vase'][0]:.2f}",
        'FullJ': f"{sF['J'][0]:.2f}", 'FullJsd': f"{sF['J'][1]:.2f}", 'FullFrac': f"{100 * sF['J_frac'][0]:.0f}", 'FullFracMin': f"{100 * min(r['final']['J'] for r in abl['full']) / opt['constrained']['J']:.0f}",
        'FullFracMax': f"{100 * max(r['final']['J'] for r in abl['full']) / opt['constrained']['J']:.0f}",
        'FullC': f"{sF['C'][0]:.3f}", 'FullMaxC': f"{sF['max_C'][0]:.3f}", 'FullInfeas': f"{100 * sF['frac_infeasible'][0]:.0f}",
        'DualInfeas': f"{100 * sD['frac_infeasible'][0]:.0f}", 'DualMaxC': f"{sD['max_C'][0]:.3f}", 'DualC': f"{sD['C'][0]:.3f}", 'DualJ': f"{sD['J'][0]:.2f}",
        'SymSwitch': f"{sS['p_switch'][0]:.2f}", 'SymJ': f"{sS['J'][0]:.2f}",
        'KSwitch': f"{summarize(abl['K'], opt)['p_switch'][0]:.2f}", 'ADam': f"{summarize(abl['A'], opt)['p_dam'][0]:.2f}",
        'ExKSwitch': f"{exact['K']['p_switch']:.2f}", 'ExADam': f"{exact['A']['p_dam']:.2f}", 'ExMVase': f"{exact['M']['p_vase']:.2f}", 'ExMSwitch': f"{exact['M']['p_switch']:.2f}",
        'ExKAC': f"{exact['KA']['C']:.3f}", 'ExSymSwitch': f"{exact['full_sym']['p_switch']:.2f}", 'ExSymJ': f"{abs(exact['full_sym']['J']):.2f}",
        'ExMNoShieldDam': f"{exact['M_noshield']['p_dam']:.3f}", 'ExKJ': f"{exact['K']['J']:.2f}", 'ExAJ': f"{exact['A']['J']:.2f}", 'ExMJ': f"{exact['M']['J']:.2f}",
        'ExAMVase': f"{exact['AM']['p_vase']:.2f}", 'ExKMSwitch': f"{exact['KM']['p_switch']:.2f}",
        'KSwitchSeeds': f"{sum(1 for r in abl['K'] if r['final']['p_switch'] > 0.5)}",
        'ADamSeeds': f"{sum(1 for r in abl['A'] if r['final']['p_dam'] > 0.5)}", 'AVaseSeeds': f"{sum(1 for r in abl['A'] if r['final']['p_vase'] > 0.5)}",
        'KMSwitchSeeds': f"{sum(1 for r in abl['KM'] if r['final']['p_switch'] > 0.5)}", 'AMVaseSeeds': f"{sum(1 for r in abl['AM'] if r['final']['p_vase'] > 0.5)}",
        'MNoShieldSwitchSeeds': f"{sum(1 for r in abl['M_noshield'] if r['final']['p_switch'] > 0.5)}",
        'BaseOptSeeds': f"{sum(1 for r in abl['none'] if r['final']['J'] > 2.3)}",
        'LateSwitchEnd': f"{summarize(loj['late'], opt)['p_switch'][0]:.2f}",
        'MVase': f"{summarize(abl['M'], opt)['p_vase'][0]:.2f}", 'MSwitch': f"{summarize(abl['M'], opt)['p_switch'][0]:.2f}",
        'KAC': f"{summarize(abl['KA'], opt)['C'][0]:.3f}",
        'DamThrLevel': f"{irr['level']['threshold']['p_dam']:.4f}", 'DamStarLevel': f"{irr['prediction']['dam']['lam_star']:.4f}",
        'DamThrOneoff': f"{irr['oneoff']['threshold']['p_dam']:.3f}", 'DamStarOneoff': f"{irr['prediction']['dam']['lam_star_oneoff']:.3f}",
        'DamRatio': f"{irr['oneoff']['threshold']['p_dam'] / irr['level']['threshold']['p_dam']:.1f}",
        'SwitchThr': f"{sw['full']['level']['threshold']['p_switch']:.3f}", 'SwitchStarMin': f"{sw['full']['prediction']['switch']['lam_star']:.2f}",
        'SwitchStarMean': f"{sw['full']['prediction']['switch']['lam_star_mean']:.3f}", 'SwitchOmegaMin': f"{sw['full']['prediction']['switch']['Omega']:.3f}",
        'SwitchOmegaMean': f"{sw['full']['prediction']['switch']['Omega_mean']:.3f}", 'SwitchG': f"{sw['full']['prediction']['switch']['G']:.2f}",
        'DamG': f"{irr['prediction']['dam']['G']:.3f}", 'DamOmega': f"{irr['prediction']['dam']['Omega']:.3f}",
        'FullDamThr': f"{sw['full']['level']['threshold']['p_dam']:.3f}", 'FullDamStar': f"{sw['full']['prediction']['dam']['lam_star']:.3f}",
        'ShiftOff': f"{pd_off:.2f}", 'ShiftOn': f"{pd_on:.2f}", 'ShiftRed': f"{100 * (1 - pd_on / pd_off):.0f}" if pd_off > 0 else '--',
        'ShiftOnFive': f"{pd_on5:.2f}", 'ShiftRedFive': f"{100 * (1 - pd_on5 / pd_off):.0f}" if pd_off > 0 else '--',
        'ShiftRetFifteenOff': f"{st['full']['off']['return'][0][i15]:.2f}", 'ShiftRetFifteenFive': f"{st['full']['on_5']['return'][0][i15]:.2f}",
        'ShiftAlarmFive': f"{st['full']['on_5']['alarm_time'][0][i15]:.0f}", 'ShiftAlarmRateFive': f"{100 * st['full']['on_5']['alarm_rate'][0][i15]:.0f}",
        'ShiftMildOffDam': f"{st['full']['off']['p_dam'][0][i2]:.3f}", 'ShiftMildFiveDam': f"{st['full']['on_5']['p_dam'][0][i2]:.3f}",
        'ShiftMildOffRet': f"{st['full']['off']['return'][0][i2]:.2f}", 'ShiftMildFiveRet': f"{st['full']['on_5']['return'][0][i2]:.2f}",
        'AdvHide': f"{sum(1 for r in loj['adversity'] if r['final']['p_switch'] > 0.5)}",
        'ShiftRetZeroOff': f"{ret_off0:.3f}", 'ShiftRetZeroOn': f"{ret_on0:.3f}",
        'ShiftAlarmFifteen': f"{st['full']['on_12']['alarm_time'][0][i15]:.0f}",
        'LateJ': f"{late['J'][0]:.2f}", 'LateProbe': f"{late['seeds_any_probe']}", 'LateInaction': f"{sum(1 for r in loj['late'] if r['final']['J'] < 0.05)}",
        'NoPrelimJ': f"{nop['J'][0]:.2f}", 'NoPrelimProbe': f"{nop['seeds_any_probe']}",
        'NoOptJ': f"{summarize(loj['no_optimism'], opt)['J'][0]:.2f}" if 'no_optimism' in loj else '--',
        'NoOptInaction': f"{sum(1 for r in loj['no_optimism'] if r['final']['J'] < 0.05)}" if 'no_optimism' in loj else '--',
        'NoOptNoPreInaction': f"{sum(1 for r in loj['no_optimism_no_prelim'] if r['final']['J'] < 0.05)}" if 'no_optimism_no_prelim' in loj else '--',
        'AdvInaction': f"{sum(1 for r in loj['adversity'] if r['final']['J'] < 0.05)}",
        'PreIters': f"{msd([r['pre_iters'] for r in abl['full']])[0]:.1f}",
        'NoShieldInaction': f"{sum(1 for r in abl['full_noshield'] if r['final']['J'] < 0.05)}",
        'LevelVase': f"{opt['levels']['vase_broken']['dK']:.3f}", 'LevelDam': f"{opt['levels']['dam_open']['dK']:.3f}",
        'LevelSwitch': f"{opt['levels']['switch_disabled']['dA']:.3f}", 'SymShutdown': f"{opt['shutdown_dA_sym']:.3f}",
        'PowerBase': f"{opt['levels']['baseline']['POWER']:.3f}", 'PowerSwitch': f"{opt['levels']['switch_disabled']['POWER']:.3f}",
        'dAswitchMax': f"{opt['dA_switch_max']:.3f}",
        'NStates': str(opt['n_states']),
    }
    if 'adversity' in st:
        M['AdvOff'] = f"{st['adversity']['off']['p_dam'][0][i15]:.2f}"; M['AdvOn'] = f"{st['adversity']['on_12']['p_dam'][0][i15]:.2f}"
        M['AdvJ'] = f"{summarize(loj['adversity'], opt)['J'][0]:.2f}"
    with open(os.path.join(ROOT, 'paper', 'numbers.tex'), 'w') as f:
        for k, v in M.items():
            f.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")
    return M


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    opt = load('optima.json'); sw = load('sweep.json'); exact = load('exact_ablation.json')
    sens = by_cond(load('sensitivity.json')) if os.path.exists(os.path.join(RES, 'sensitivity.json')) else {}
    abl = by_cond(load('ablation.json')); loj = by_cond(load('lojong.json')); sh = load('shift.json')
    fig_map(); fig_architecture(); fig_main(abl, opt); fig_ablation(abl, opt); fig_sweep(sw); fig_multiplier(abl); fig_lojong(abl, loj, opt); fig_shift(sh)
    tab_main(abl, opt); tab_ablation(abl, opt, exact); tab_sweep(sw); tab_sensitivity(abl, sens, opt); tab_lojong(abl, loj, opt); tab_shift(sh)
    M = numbers(abl, loj, opt, sw, sh, exact)
    for k, v in M.items():
        print(f'{k:16s} {v}')
