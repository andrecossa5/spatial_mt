"""
05 - Diagnostics of the final character set: is it real mtDNA signal, and how weak are the weakest calls?

    python code/05_diagnostics.py

Writes results/diagnostics.txt (also printed). Compares the characters' carrier calls with the background of all
other alt calls (spectrum, strand and read-position statistics), counts the characters seen in several sequencing runs,
and counts weak calls (1-2 alt reads, shallow cells). Uses the upstream tables for strand / read-position columns.
"""

import io
import logging
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from common import DATA, PREFIX, RESULTS, TRANSITIONS, final_afm_and_tree, make_dirs

logging.basicConfig(level=logging.WARNING)


def report():
    afm, _ = final_afm_and_tree()
    allelic = pd.read_csv(DATA / f'{PREFIX}allelic_table.tsv.gz', sep='\t')
    annotated = pd.read_csv(
        DATA / f'{PREFIX}annotated_allelic_table.tsv.gz', sep='\t',
        usecols=['sample', 'POS', 'ALT', 'MUT_TYPE', 'strand', 'AF'],
    )
    allelic = allelic.merge(annotated, on=['sample', 'POS', 'ALT'])
    allelic['var'] = allelic['POS'].astype(str) + '_' + allelic['REF'] + '>' + allelic['ALT']

    B = afm.layers['bin'].toarray() > 0
    AD, DP, AF = afm.layers['AD'].toarray(), np.asarray(afm.layers['DP']), afm.X.toarray()
    print(f'{afm.shape[0]} cells x {afm.shape[1]} characters; {int(B.sum())} carrier calls')

    # carrier calls vs the background of every other alt call
    carriers = pd.DataFrame(
        [(afm.obs_names[i], v) for j, v in enumerate(afm.var_names) for i in np.flatnonzero(B[:, j])],
        columns=['sample', 'var'],
    ).merge(allelic, on=['sample', 'var'], how='left')
    background = allelic[~allelic['var'].isin(afm.var_names)].sample(300000, random_state=0)

    def summary(d, name):
        print(f'{name}: n={len(d)} | AF median {d.AF.median():.3f} | total_count median {d.total_count.median():.0f} | '
              f'sb_pval<.05 {np.mean(d.sb_pval < .05):.3f} | pos_end_read median {d.pos_end_read.median():.0f}, '
              f'<15: {np.mean(d.pos_end_read < 15):.3f} | overlap fraction {np.mean(d.overlap_count / d.total_count):.3f}')

    summary(carriers, 'carrier calls')
    summary(background, 'background alt calls')
    print('carrier AF quantiles (5/25/50/75/95%):', carriers.AF.quantile([.05, .25, .5, .75, .95]).round(3).to_dict())

    sub = (afm.var['ref'].astype(str) + '>' + afm.var['alt'].astype(str))
    print(f'transition fraction: characters {sub.isin(TRANSITIONS).mean():.2f}, '
          f'background alt calls {(background.REF + ">" + background.ALT).isin(TRANSITIONS).mean():.2f}')
    print('characters by substitution:', sub.value_counts().to_dict())
    carriers['run'] = carriers['sample'].str.split('-').str[0]
    print('characters by number of sequencing runs carrying them:',
          carriers.groupby('var').run.nunique().value_counts().sort_index().to_dict())

    # weak calls
    ad, dp, af = AD[B], DP[B], AF[B]
    print(f'weak calls among {B.sum()} carrier calls: AD==1 {np.mean(ad == 1):.1%}, AD<=2 {np.mean(ad <= 2):.1%}, '
          f'DP<50 {np.mean(dp < 50):.1%}, DP<100 {np.mean(dp < 100):.1%}, AF<0.01 {np.mean(af < .01):.1%}')
    print(f'colour scale (vmax 0.1) clips {np.mean(af > .1):.0%} of the carrier calls; '
          f'AF quantiles (5/25/50/75/95%): {np.quantile(af, [.05, .25, .5, .75, .95]).round(3).tolist()}')


if __name__ == '__main__':
    make_dirs()
    buf = io.StringIO()
    with redirect_stdout(buf):
        report()
    text = buf.getvalue()
    print(text)
    (RESULTS / 'diagnostics.txt').write_text(text)
