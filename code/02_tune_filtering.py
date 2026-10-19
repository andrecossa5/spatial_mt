"""
02 - Tune the variant filtering, with a permutation null to calibrate it.

    python code/02_tune_filtering.py [--seeds 5] [--skip-null]

Two tables, in results/:
  tuning_sweep.csv  one parameter changed at a time from the filter_afm defaults (counts and quality proxies)
  tuning_null.csv   (geno_alpha, qc_alpha) grid; each setting is run on the data and on permuted AFMs
                    (per-variant AD/DP shuffled across cells), whose characters are false discoveries:
                    fdr = null characters / real characters, net_* = real - null.

Quality proxies for the sweep: fraction of transitions among the characters, fraction carried in >= 2 sequencing
runs, and fraction in the upstream pipeline's own `confident` set. Replication across runs becomes uninformative
at loose genotyping thresholds (calls spread over runs trivially), which is why the null is used to choose.
"""

import argparse
import logging
import time

import numpy as np
import pandas as pd
from common import (
    RESULTS, character_metrics, load_confident_muts, load_raw_afm, make_dirs, permuted, run_filter_afm, select_cells,
)

logging.basicConfig(level=logging.WARNING)

# (label, cell selection, filter_afm overrides); defaults of filter_afm elsewhere, uncapped candidates
SWEEP = [('baseline', dict(), {})]
SWEEP += [(f'mincov>{c}', dict(min_cov=c), {}) for c in (0, 1, 3, 5)]
SWEEP += [('no upper cell cutoff', dict(upper_q=None), {})]
SWEEP += [(f'cand_q_max={q}', {}, dict(cand_q_max=q)) for q in (0.1, 0.2, 0.5)]
SWEEP += [(f'qc_alpha={q}', {}, dict(qc_alpha=q)) for q in (0.1, 0.2, 0.5)]
SWEEP += [(f'geno_alpha={g}', {}, dict(geno_alpha=g)) for g in (1e-2, 1e-4, 1e-5)]
SWEEP += [('impute', {}, dict(impute=True)), ('max_prevalence=.9', {}, dict(max_prevalence=.9))]

NULL_GRID = [
    (1e-4, .05), (1e-4, .1), (1e-3, .02), (1e-3, .05), (1e-3, .1), (1e-3, .2),
    (3e-3, .02), (3e-3, .05), (3e-3, .1), (5e-3, .1), (5e-3, .2), (7e-3, .1), (1e-2, .05), (1e-2, .1),
]
BASE = dict(geno_alpha=1e-3, qc_alpha=.05)      # filter_afm defaults, for the sweep


def run_sweep(raw, confident):
    rows = []
    for label, cells, over in SWEEP:
        afm = select_cells(raw, **cells)
        t = time.time()
        try:
            f = run_filter_afm(afm, **{**BASE, **over})
            m = character_metrics(f, afm.shape[0], confident)
        except ValueError as e:
            m = dict(cells_in=afm.shape[0], error=str(e)[:60])
        rows.append({'label': label, **m, 'seconds': round(time.time() - t)})
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def run_null(raw, n_seeds):
    afm = select_cells(raw)
    nulls = [permuted(afm, s) for s in range(1, n_seeds + 1)]
    rows = []
    for g, q in NULL_GRID:
        real = run_filter_afm(afm, geno_alpha=g, qc_alpha=q)
        nc, nn = [], []
        for nb in nulls:
            try:
                f = run_filter_afm(nb, geno_alpha=g, qc_alpha=q)
                nc.append(f.shape[1])
                nn.append(f.shape[0])
            except ValueError:          # nothing survives on this null
                nc.append(0)
                nn.append(0)
        rows.append(dict(
            geno_alpha=g, qc_alpha=q, chars=real.shape[1], cells=real.shape[0],
            null_chars=np.mean(nc), null_cells=np.mean(nn),
            net_chars=round(real.shape[1] - np.mean(nc), 1), net_cells=round(real.shape[0] - np.mean(nn)),
            fdr=round(np.mean(nc) / real.shape[1], 3),
        ))
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=5, help='permutations per setting of the null')
    p.add_argument('--skip-null', action='store_true')
    args = p.parse_args()

    make_dirs()
    raw = load_raw_afm()
    run_sweep(raw, load_confident_muts()).to_csv(RESULTS / 'tuning_sweep.csv', index=False)
    if not args.skip_null:
        run_null(raw, args.seeds).to_csv(RESULTS / 'tuning_null.csv', index=False)
