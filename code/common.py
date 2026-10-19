"""
Shared helpers for the spatial_mt analysis: paths, tables -> AFM, cell filter, the chosen
filter_afm settings, the permutation null, and quality metrics.

Data are single cells (PTA) of normal colon, pre-processed outside MiTo into long tables
(allelic table, coverage table, mt copy number), so the AFM is built with
`mito.io.make_afm_from_counts` rather than one of the standard readers.

Paths are relative to the repository. Override with the environment variables
SPATIAL_MT_DATA (input tables) and SPATIAL_MT_OUT (parent of `results/` and `figures/`).
"""

import os
from pathlib import Path

import anndata as ad
import mito as mt
import numpy as np
import pandas as pd
import scipy.sparse as sp

assert mt.__version__ == '2.0.0', f'MiTo 2.0.0 expected, found {mt.__version__} (see envs/environment.yml)'

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('SPATIAL_MT_DATA', ROOT / 'data' / 'all_normal_colon_deepseq'))
OUT = Path(os.environ.get('SPATIAL_MT_OUT', ROOT))
RESULTS, FIGURES = OUT / 'results', OUT / 'figures'
PREFIX = 'all_normal_colon_deepseq.'

# Choices made in the analysis (see 02_tune_filtering.py and 03_clones.py for how)
CELL_MIN_COV = 10        # colonies/cells with mean MT coverage above this ...
CELL_UPPER_Q = 0.99      # ... and not above this quantile of mean MT coverage
FINAL_FILTER = dict(max_candidates=None, geno_alpha=1e-2, qc_alpha=0.05)
FINAL_TAU = 0            # annotate_clones: no neighbourhood abstention (it fragments clones, see 03_clones.py)

TRANSITIONS = {'A>G', 'G>A', 'C>T', 'T>C'}


def make_dirs():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


##


def read_tables():
    """Allelic table (alt molecules), coverage table and MT copy number, as written by the upstream pipeline."""
    allelic = pd.read_csv(DATA / f'{PREFIX}allelic_table.tsv.gz', sep='\t')
    coverage = pd.read_csv(DATA / f'{PREFIX}coverage_table.tsv.gz', sep='\t')
    mt_cn = pd.read_csv(DATA / f'{PREFIX}mt_cn.tsv.gz', sep='\t').set_index('colony_name')
    return allelic, coverage, mt_cn


def build_afm():
    """
    AFM from the tables. Mapping onto the MiTo contract:
      AD = total_count          (alt fragments; a read pair counts once: total = fw + rev + overlap)
      DP = callable_coverage    (what the pipeline's own AF is computed on: AF = total_count / callable_coverage)
    scLT_system 'ReDeeM' is the untargeted DNA path: no gene restriction, no RNA-edit filter.
    """
    allelic, coverage, mt_cn = read_tables()
    counts = allelic.rename(
        columns={'sample': 'cell', 'POS': 'pos', 'ALT': 'alt', 'REF': 'ref', 'total_count': 'AD'}
    )[['cell', 'pos', 'ref', 'alt', 'AD']]
    cov = coverage.rename(
        columns={'sample': 'cell', 'POS': 'pos', 'callable_coverage': 'DP'}
    )[['cell', 'pos', 'DP']]

    cells = pd.Index(pd.unique(cov['cell']))
    meta = mt_cn.loc[cells].copy()
    meta['run'] = cells.str.split('-').str[0].values                       # sequencing run
    meta['plate'] = (
        cells.str.split('-').str[1].str.replace(r'_[A-Z]-?\d.*|_W.*', '', regex=True).values
    )
    return mt.io.make_afm_from_counts(
        counts, cov, cell_meta=meta, scLT_system='ReDeeM', pp_method='colon_deepseq_tables'
    )


def load_raw_afm():
    """The raw AFM written by 01_build_afm.py (built on the fly if it is not there)."""
    path = RESULTS / 'afm_raw.h5ad'
    return ad.read_h5ad(path) if path.exists() else build_afm()


def load_confident_muts():
    """Mutations the upstream pipeline itself called confident, as MiTo variant names (pos_ref>alt)."""
    conf = pd.read_csv(DATA / f'{PREFIX}confident_allelic_table.tsv.gz', sep='\t')
    return set(conf['MUT'].str.replace(r'^(\d+)_(\w)_(\w)$', r'\1_\2>\3', regex=True))


##


def select_cells(raw, min_cov=CELL_MIN_COV, upper_q=CELL_UPPER_Q):
    """Cells with mean MT coverage > min_cov and <= the upper_q quantile (None: no upper bound)."""
    cov = raw.obs['mean_site_coverage']
    keep = cov > min_cov
    if upper_q is not None:
        keep &= cov <= cov.quantile(upper_q)
    afm = raw[keep.values].copy()
    afm._inplace_subset_var((afm.X > 0).sum(0).A1 > 0)       # sites observed in at least one cell
    return afm


def run_filter_afm(afm, copy=True, **overrides):
    """filter_afm with the chosen settings (FINAL_FILTER), optionally overridden."""
    kw = {**FINAL_FILTER, **overrides}
    return mt.pp.filter_afm(afm, copy=copy, ncores=4, **kw)


def final_afm_and_tree(raw=None):
    """Chosen cells -> filter_afm -> UPGMA tree -> clones. Returns (afm with MiTo_clone, tree)."""
    raw = load_raw_afm() if raw is None else raw
    afm = select_cells(raw)
    mt.pp.filter_afm(afm, ncores=4, **FINAL_FILTER)
    tree = mt.tl.build_tree(afm)
    mt.tl.annotate_clones(tree, afm, tau=FINAL_TAU)
    return afm, tree


##


def permuted(afm, seed):
    """
    Null AFM: each variant's AD and DP are permuted across cells together (per-variant depth and AD marginals
    and their pairing kept; clonal structure destroyed). Whatever filter_afm keeps on it is a false discovery.
    """
    rng = np.random.default_rng(seed)
    out = afm.copy()
    AD = out.layers['AD'].toarray()
    DP = np.asarray(out.layers['DP']).copy()
    for j in range(AD.shape[1]):
        p = rng.permutation(AD.shape[0])
        AD[:, j] = AD[p, j]
        DP[:, j] = DP[p, j]
    out.layers['AD'] = sp.csr_matrix(AD)
    out.layers['DP'] = DP
    out.X = sp.csr_matrix((AD / np.maximum(DP, 1)).astype(np.float32))
    return out


def character_metrics(f, n_cells_in, confident):
    """Quality proxies of a filtered AFM: counts, transition fraction, replication across runs, overlap with `confident`."""
    B = f.layers['bin'].toarray() > 0
    n = B.sum(1)
    runs = pd.Series(f.obs['run'].values)
    n_runs = np.array([runs[B[:, j]].nunique() for j in range(B.shape[1])])
    sub = f.var['ref'].astype(str) + '>' + f.var['alt'].astype(str)
    return dict(
        cells_in=n_cells_in, cells=int((n > 0).sum()), chars=int(f.shape[1]),
        per_cell=round(float(n[n > 0].mean()), 2),
        ti=round(float(sub.isin(TRANSITIONS).mean()), 2),
        multi_run=round(float((n_runs >= 2).mean()), 2),
        in_conf=round(float(np.mean([v in confident for v in f.var_names])), 2),
        min_prev=int(B.sum(0).min()),
    )
