# Bodhi-Regularization v2 — code, results and arXiv sources

Lurong Pan, *Bodhi-Regularization: A Dharmic Dynamics Framework for Alignment via Global Regularization*, v2 (October 2026).

```
code/env.py              Bodhi Gridworld: tabular MDP with four behavioural probes (vase, dam, field, off-switch)
code/critics.py          Karma (binary relative reachability), Anatta (one-sided excess POWER), suffering;
                         exact evaluation, value iteration, occupancy-measure LP for the constrained optimum
code/brag.py             B-RAG: PPO on the Bodhi objective with dual-ascent or log-barrier Metta constraint,
                         pre-action shield, preliminaries phase, lojong protocol options
code/monitor.py          Samprajanya: CUSUM surprise detector + retreat-and-wait fallback
code/run_experiments.py  the experiment suite  (writes results/*.json, results/policies/*.npy)
code/analyze.py          figures and LaTeX tables (writes paper/figs, paper/tables, paper/numbers.tex)
tests/test_brag.py       unit tests: kernel, critic levels, exact optima, exact-vs-Monte-Carlo, Proposition 3 (1 min)
paper/                   LaTeX sources (main.tex + sections), refs.bib, figures, tables
build_arxiv.py           compiles the paper, builds dist/arxiv_submission.tar.gz, test-compiles it clean-room
results/                 raw results (JSON) of the runs reported in the paper
```

## Reproduce everything

Requirements: Python 3 with NumPy, SciPy and Matplotlib; a TeX Live with pdflatex and bibtex
(packages: natbib, booktabs, algorithm/algpseudocode, hyperref, microtype, caption, enumitem).

```bash
python3 build_arxiv.py --run        # experiments (~30 min on 2 cores) + figures + paper + arXiv tarball
BRAG_QUICK=1 python3 build_arxiv.py --run   # 2 seeds x 120 iterations per condition (~3 min), for a quick look
python3 -m pytest tests -q          # unit tests (~1 min)
python3 build_arxiv.py --analyze    # regenerate figures/tables from existing results, then build
python3 build_arxiv.py              # just compile and package
python3 build_arxiv.py --docx       # additionally produce a Word version (pandoc)
```

Or step by step:

```bash
cd code
python3 run_experiments.py optima exact_ablation ablation lojong sensitivity sweep shift
python3 analyze.py
cd ../paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Submitting to arXiv

Upload `dist/arxiv_submission.tar.gz` as the source. It contains `main.tex` at top level, all `\input`
files, the pre-built `main.bbl`, PDF figures, and an `anc/` directory with the code and results that arXiv
lists as ancillary files. Suggested primary category: cs.AI (cross-list cs.LG).

All reported returns, constraint costs and probe probabilities are exact dynamic-programming values of the
trained tabular policies; `paper/numbers.tex` is generated from the results so the text cannot drift from the data.
