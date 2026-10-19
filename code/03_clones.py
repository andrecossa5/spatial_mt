"""
03 - Tree and clones: sweep annotate_clones (tau, min_support, rescue), then write the chosen result.

    python code/03_clones.py

Writes to results/: afm_final.h5ad (filtered AFM with MiTo_clone), clones.csv (size, clade support and markers per
clone), tree.newick, clone_sweep.csv.

Why tau = 0: the default neighbourhood abstention (tau = 0.25) blanks out cells inside a clone's clade, which cuts
single-label clades into pieces (10 of 22 clones drawn in several pieces) and leaves 64 cells unassigned. With
tau = 0 clones are unsplit and 277 of 294 cells are assigned (96% carry >= 1 marker of their clone). `rescue` adds
1-6 cells in every setting; `min_support` trades many cells for few weak clones.
"""

import itertools
import logging

import mito as mt
import numpy as np
import pandas as pd
from common import FINAL_TAU, RESULTS, final_afm_and_tree, load_raw_afm, make_dirs, run_filter_afm, select_cells

logging.basicConfig(level=logging.WARNING)


def single_label_pieces(tree, labels, clone):
    """Maximal clades of the tree whose cells all carry this clone label (1 piece = a clone drawn as one block)."""
    pieces = []

    def rec(node):
        leaves = tree.leaves_in_subtree(node)
        present = {labels[c] for c in leaves}
        if present == {clone}:
            pieces.append(leaves)
        elif clone in present:
            for child in tree.children(node):
                rec(child)

    rec(tree.root)
    return pieces


def evaluate(afm, tree):
    """Assigned cells, clones, fragmentation and marker concordance of the current MiTo_clone annotation."""
    lab = afm.obs['MiTo_clone'].astype(str)
    labels = lab.to_dict()
    clones = afm.uns['mito']['annotate_clones']['clones']
    B = pd.DataFrame(afm.layers['bin'].toarray() > 0, index=afm.obs_names, columns=afm.var_names)
    assigned = lab != 'unassigned'
    split = stray = 0
    concordant = []
    for c in clones.index:
        cells = lab.index[lab == c]
        sizes = [len(p) for p in single_label_pieces(tree, labels, c)]
        split += len(sizes) > 1
        stray += len(cells) - max(sizes)
        markers = [m for m in clones.loc[c, 'markers'].split(';') if m]
        if markers:
            concordant += list(B.loc[cells, markers].any(axis=1))
    return dict(
        assigned=int(assigned.sum()), clones=int(lab[assigned].nunique()),
        clones_ge5=int((lab[assigned].value_counts() >= 5).sum()), split=int(split), stray_cells=int(stray),
        marker_concordance=round(float(np.mean(concordant)), 2),
        median_support=round(float(clones['support'].dropna().median()), 2),
    )


if __name__ == '__main__':
    make_dirs()
    raw = load_raw_afm()

    # 1. sweep, on a fixed tree (clade support is computed once)
    base = run_filter_afm(select_cells(raw), copy=True)
    tree = mt.tl.build_tree(base)
    mt.tl.annotate_clones(tree, base)
    rows = []
    for tau, min_support, rescue in itertools.product((0.25, 0.1, 0.0), (None, 0.3, 0.5), (False, True)):
        afm = base.copy()
        mt.tl.annotate_clones(tree, afm, tau=tau, min_support=min_support, rescue=rescue, support=False)
        rows.append(dict(tau=tau, min_support=min_support, rescue=rescue, **evaluate(afm, tree)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / 'clone_sweep.csv', index=False)

    # 2. chosen: tau = FINAL_TAU, no min_support, no rescue
    afm, tree = final_afm_and_tree(raw)
    clones = afm.uns['mito']['annotate_clones']['clones']
    lab = afm.obs['MiTo_clone'].astype(str)
    print(f'tau={FINAL_TAU}: {afm.shape[0]} cells x {afm.shape[1]} variants; '
          f'{(lab != "unassigned").sum()} cells assigned to {lab[lab != "unassigned"].nunique()} clones')
    print(clones[['n_cells', 'support', 'markers']].to_string())
    n_markers = clones['markers'].fillna('').map(lambda s: len([x for x in s.split(';') if x]))
    print('markers per clone:', n_markers.value_counts().sort_index().to_dict())

    clones.to_csv(RESULTS / 'clones.csv')
    mt.io.write_newick(tree, str(RESULTS / 'tree.newick'))
    afm.write_h5ad(RESULTS / 'afm_final.h5ad')
