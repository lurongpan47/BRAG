#!/usr/bin/env python3
"""
Build the arXiv submission package for "Bodhi-Regularization" (v2).

    python3 build_arxiv.py               # compile the paper and package it (results must exist)
    python3 build_arxiv.py --run         # also run the full experiment suite first (~20 min, 2 cores)
    python3 build_arxiv.py --analyze     # regenerate figures/tables from results before compiling
    python3 build_arxiv.py --docx        # also produce a Word version with pandoc (needs pandoc)
    python3 build_arxiv.py --no-verify   # skip the clean-room test compile of the tarball

Outputs (in dist/):
    Bodhi-Regularization-v2.pdf          the compiled paper
    arxiv_submission.tar.gz              LaTeX sources + .bbl + figures (+ code and results as anc/ ancillary files)
    Bodhi-Regularization-v2.docx         (with --docx)
    brag_v2_code_and_results.zip         the full reproducibility bundle

arXiv notes: the tarball contains main.tex at the top level, all \\input files, the pre-built main.bbl
(arXiv runs pdflatex but not always bibtex, so the .bbl must be shipped), PDF figures, and an anc/
directory of ancillary files, which arXiv lists alongside the paper without compiling them.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER, CODE, RESULTS, DIST = (os.path.join(ROOT, d) for d in ('paper', 'code', 'results', 'dist'))
MAIN = 'main'
PDF_NAME = 'Bodhi-Regularization-v2'


def sh(cmd, cwd, quiet=True, check=True):
    r = subprocess.run(cmd, cwd=cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if check and r.returncode != 0:
        print(r.stdout[-4000:])
        raise SystemExit(f'command failed: {cmd}')
    if not quiet:
        print(r.stdout)
    return r.stdout


def run_experiments():
    sh(f'{sys.executable} run_experiments.py', cwd=CODE, quiet=False)


def analyze():
    sh(f'{sys.executable} analyze.py', cwd=CODE, quiet=False)


def compile_latex(workdir, bibtex=True):
    """pdflatex -> bibtex -> pdflatex x2; fails on LaTeX errors or unresolved references."""
    pdflatex = f'pdflatex -interaction=nonstopmode -halt-on-error {MAIN}.tex'
    sh(pdflatex, workdir)
    if bibtex:
        sh(f'bibtex {MAIN}', workdir)
    sh(pdflatex, workdir)
    log = sh(pdflatex, workdir)
    log_text = open(os.path.join(workdir, f'{MAIN}.log'), errors='ignore').read()
    problems = [l for l in log_text.splitlines() if 'undefined' in l.lower() and ('reference' in l.lower() or 'citation' in l.lower())]
    if problems:
        print('\n'.join(problems[:20]))
        raise SystemExit('unresolved references or citations')
    pages = re.search(r'Output written on .*\((\d+) pages', log_text)
    return int(pages.group(1)) if pages else None


def package(pages):
    os.makedirs(DIST, exist_ok=True)
    shutil.copy(os.path.join(PAPER, f'{MAIN}.pdf'), os.path.join(DIST, f'{PDF_NAME}.pdf'))
    stage = tempfile.mkdtemp(prefix='arxiv_')
    # sources
    for pattern in ('*.tex', f'{MAIN}.bbl'):
        for f in glob.glob(os.path.join(PAPER, pattern)):
            shutil.copy(f, stage)
    os.makedirs(os.path.join(stage, 'tables')); os.makedirs(os.path.join(stage, 'figs'))
    for f in glob.glob(os.path.join(PAPER, 'tables', '*.tex')):
        shutil.copy(f, os.path.join(stage, 'tables'))
    tex_all = ''.join(open(f).read() for f in glob.glob(os.path.join(PAPER, '*.tex')))
    used = set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{figs/([^}]+)\}', tex_all))
    for f in glob.glob(os.path.join(PAPER, 'figs', '*.pdf')):
        if os.path.basename(f) in used:            # ship only the figures the paper uses
            shutil.copy(f, os.path.join(stage, 'figs'))
    # ancillary files: code and results, so the paper is reproducible from the arXiv listing itself
    anc = os.path.join(stage, 'anc'); os.makedirs(os.path.join(anc, 'code')); os.makedirs(os.path.join(anc, 'results'))
    for f in glob.glob(os.path.join(CODE, '*.py')):
        shutil.copy(f, os.path.join(anc, 'code'))
    for f in glob.glob(os.path.join(RESULTS, '*.json')):
        shutil.copy(f, os.path.join(anc, 'results'))
    shutil.copy(os.path.join(ROOT, 'README.md'), anc)
    shutil.copy(os.path.join(ROOT, 'build_arxiv.py'), anc)
    shutil.copy(os.path.join(ROOT, 'requirements.txt'), anc)
    os.makedirs(os.path.join(anc, 'tests')); shutil.copy(os.path.join(ROOT, 'tests', 'test_brag.py'), os.path.join(anc, 'tests'))
    with open(os.path.join(stage, '00README.txt'), 'w') as f:
        f.write('Bodhi-Regularization v2 -- arXiv source.\n'
                f'Main file: {MAIN}.tex (pdflatex; bibliography pre-built in {MAIN}.bbl).\n'
                'anc/ holds the code and raw results that reproduce every figure and number.\n')
    out = os.path.join(DIST, 'arxiv_submission.tar.gz')
    with tarfile.open(out, 'w:gz') as tar:
        for name in sorted(os.listdir(stage)):
            tar.add(os.path.join(stage, name), arcname=name)
    print(f'arXiv package: {out}  ({os.path.getsize(out) / 1e6:.2f} MB, paper {pages} pages)')
    return stage, out


def verify(tarball, pages):
    """Clean-room test: extract the tarball and compile it with pdflatex only (no bibtex), as arXiv would."""
    tmp = tempfile.mkdtemp(prefix='arxiv_verify_')
    with tarfile.open(tarball) as tar:
        tar.extractall(tmp, filter='data') if hasattr(tarfile, 'data_filter') else tar.extractall(tmp)
    got = compile_latex(tmp, bibtex=False)
    if got != pages:
        raise SystemExit(f'clean-room compile produced {got} pages, expected {pages}')
    print(f'clean-room compile OK ({got} pages)')


def bundle_code():
    out = os.path.join(DIST, 'brag_v2_code_and_results.zip')
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for base in (CODE, RESULTS, PAPER):
            for dirpath, _, files in os.walk(base):
                if '__pycache__' in dirpath or 'policies' in dirpath:
                    continue
                for fn in files:
                    if fn.endswith(('.aux', '.log', '.out', '.blg', '.toc', '.synctex.gz')):
                        continue
                    full = os.path.join(dirpath, fn)
                    z.write(full, os.path.relpath(full, ROOT))
        for fn in ('README.md', 'build_arxiv.py', 'requirements.txt', os.path.join('tests', 'test_brag.py')):
            z.write(os.path.join(ROOT, fn), fn)
    print(f'code bundle: {out}  ({os.path.getsize(out) / 1e6:.2f} MB)')


def docx():
    """Word version via pandoc: PDF figures swapped for PNG renderings, citations resolved from refs.bib."""
    if shutil.which('pandoc') is None:
        print('pandoc not found; skipping .docx'); return
    tmp = tempfile.mkdtemp(prefix='docx_')
    for f in glob.glob(os.path.join(PAPER, '*.tex')) + glob.glob(os.path.join(PAPER, '*.bib')):
        shutil.copy(f, tmp)
    shutil.copytree(os.path.join(PAPER, 'tables'), os.path.join(tmp, 'tables'))
    shutil.copytree(os.path.join(PAPER, 'figs'), os.path.join(tmp, 'figs'))
    # rasterize figures for Word
    try:
        from pdf2image import convert_from_path  # optional
        for f in glob.glob(os.path.join(tmp, 'figs', '*.pdf')):
            convert_from_path(f, dpi=250)[0].save(f[:-4] + '.png')
    except Exception:
        for f in glob.glob(os.path.join(tmp, 'figs', '*.pdf')):
            sh(f'pdftoppm -png -r 250 -singlefile "{f}" "{f[:-4]}"', tmp, check=False)
    for f in glob.glob(os.path.join(tmp, '*.tex')):
        t = open(f).read().replace('.pdf}', '.png}')
        open(f, 'w').write(t)
    out = os.path.join(DIST, f'{PDF_NAME}.docx')
    sh(f'pandoc {MAIN}.tex -o "{out}" --citeproc --bibliography refs.bib --resource-path=.:figs --mathml 2>/dev/null || '
       f'pandoc {MAIN}.tex -o "{out}" --citeproc --bibliography refs.bib --resource-path=.:figs', tmp, check=False)
    if os.path.exists(out):
        print(f'Word version: {out}  ({os.path.getsize(out) / 1e6:.2f} MB)')
    else:
        print('pandoc failed to produce the .docx')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true'); ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--docx', action='store_true'); ap.add_argument('--no-verify', action='store_true')
    args = ap.parse_args()
    if args.run:
        run_experiments(); args.analyze = True
    if args.analyze:
        analyze()
    pages = compile_latex(PAPER)
    stage, tarball = package(pages)
    if not args.no_verify:
        verify(tarball, pages)
    bundle_code()
    if args.docx:
        docx()
