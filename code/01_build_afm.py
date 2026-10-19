"""
01 - Build the raw AFM from the upstream tables and check it against them.

    python code/01_build_afm.py

Writes results/afm_raw.h5ad and results/cells_coverage.csv.
"""

import logging

import numpy as np
import pandas as pd
from common import RESULTS, build_afm, make_dirs, read_tables

logging.basicConfig(level=logging.WARNING)


def spot_check(afm, n=2000, seed=0):
    """Compare random alt calls in the AFM with the tables (independently of MiTo). Returns the number of mismatches."""
    allelic, coverage, _ = read_tables()
    cov = coverage.set_index(['sample', 'POS'])['callable_coverage']
    rng = np.random.default_rng(seed)
    bad = checked = 0
    for i in rng.choice(len(allelic), n, replace=False):
        r = allelic.iloc[i]
        v = f'{r.POS}_{r.REF}>{r.ALT}'
        if v not in afm.var_names:       # not the dominant alt allele at its site: dropped by design
            continue
        checked += 1
        ci, vi = afm.obs_names.get_loc(r['sample']), afm.var_names.get_loc(v)
        ad, dp = afm.layers['AD'][ci, vi], afm.layers['DP'][ci, vi]
        if ad != r.total_count or dp != max(cov[(r['sample'], r.POS)], r.total_count):
            bad += 1
    return bad, checked


if __name__ == '__main__':
    make_dirs()
    afm = build_afm()
    print(afm)
    bad, checked = spot_check(afm)
    print(f'spot check: {bad} mismatches in {checked} calls')
    assert bad == 0

    cov = afm.obs['mean_site_coverage']
    print('mean MT coverage per cell, quantiles:', cov.quantile([0, .05, .25, .5, .75, .95, 1]).round(1).to_dict())
    afm.obs[['mean_site_coverage', 'mt_cn', 'nuclear_coverage', 'run', 'plate']].to_csv(RESULTS / 'cells_coverage.csv')
    afm.write_h5ad(RESULTS / 'afm_raw.h5ad')
