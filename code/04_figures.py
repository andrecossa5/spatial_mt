"""
04 - Figures: collapsed cell tree (oriented down) over the allele-frequency matrix, and the clone tree.

    python code/04_figures.py

Writes to figures/: tree_heatmap.pdf (vector), tree_heatmap{,_1000dpi,_2400dpi}.png, clone_tree.{pdf,png}.

Style: plotting_utils' Nature Methods rcParams, sizes scaled to a 3.5-inch panel. Matrix cells are drawn as vector
rectangles, each extended slightly right and down so that the next one covers the overlap: strokes around cells bleed
across their edges in a PDF, and no stroke leaves hairline seams. Tree and matrix share the same columns: the matrix
columns are the leaf positions of the *collapsed* tree (the layout rebuilds the tree, so its leaf order can differ from
the original tree's). `_collapse_clones`, `_place_tree` and `_get_muts_order` are MiTo internals.
"""

import logging

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mito as mt
import numpy as np
import pandas as pd
import plotting_utils as plu
from common import FIGURES, final_afm_and_tree, make_dirs
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from mito.pl.phylo_plots import _collapse_clones, _place_tree
from mito.tl.phylo import _get_muts_order

logging.basicConfig(level=logging.WARNING)

plu.set_rcParams({
    'pdf.fonttype': 42, 'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6,
    'ytick.labelsize': 6, 'legend.fontsize': 6, 'lines.linewidth': .5, 'xtick.major.size': 2, 'ytick.major.size': 2,
})


def leaf_layout(tree):
    """Leaf x positions, and the y of leaves and root, of the collapsed tree exactly as plot_tree draws it."""
    ctree, _ = _collapse_clones(tree, 'MiTo_clone', 'raw')
    leaf_length = .04 * max(ctree.get_time(leaf) for leaf in ctree.leaves)       # plot_tree's default
    _, _, _, raw = _place_tree(ctree, [], None, None, 'raw', 'down', False, True, False, leaf_length=leaf_length)
    x = pd.Series({k: v[0] for k, v in raw.items() if ctree.is_leaf(k)}).sort_values()
    assert np.allclose(np.diff(x.values), 1.0), 'non-uniform leaf spacing'
    ys = np.array([v[1] for v in raw.values()])
    return x, ys.min(), ys.max()


def tree_heatmap(afm, tree):
    lab = afm.obs['MiTo_clone'].astype(str)
    assigned = lab != 'unassigned'
    n_clones = lab[assigned].nunique()

    lx, y_leaf, y_root = leaf_layout(tree)
    cells, muts = list(lx.index), _get_muts_order(tree)
    n = len(muts)
    X = pd.DataFrame(afm.X.toarray(), index=afm.obs_names, columns=afm.var_names).loc[cells, muts].T.values
    xe = np.r_[lx.values - .5, lx.values[-1] + .5]

    W, H = 3.5, 3.2
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[.86, 1.74], width_ratios=[1, .035], hspace=.012, wspace=.04,
        left=.135, right=.87, bottom=.17 / H, top=1 - .32 / H,
    )
    ax_tree, ax_mat, ax_cb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    mt.pl.plot_tree(
        tree, ax=ax_tree, orient='down', collapse_clones='MiTo_clone', clone_style='full',
        branches=dict(annot='MiTo_clone', linewidth=.3),
    )
    xlim = (xe[0], xe[-1])
    ax_tree.set_aspect('auto')
    ax_tree.set_xlim(xlim)
    ax_tree.set_ylim(y_leaf, y_root + .01 * (y_root - y_leaf))
    ax_tree.set_title(f'{afm.shape[0]} cells ({int(assigned.sum())} assigned), {afm.shape[1]} variants, {n_clones} clones', pad=9)

    cmap, norm, e = plt.get_cmap('afmhot_r'), plt.Normalize(0, .1), .15
    verts = [[(xe[j], i), (xe[j + 1] + e, i), (xe[j + 1] + e, i + 1 + e), (xe[j], i + 1 + e)]
             for i in range(n) for j in range(X.shape[1])]
    ax_mat.add_collection(PolyCollection(
        verts, facecolors=cmap(norm(X.ravel())), edgecolors='none', linewidths=0, antialiaseds=False))
    ax_mat.set_xlim(xlim)
    ax_mat.set_ylim(n, 0)
    ax_mat.set_xticks([])
    ax_mat.set_yticks(np.arange(n) + .5)
    ax_mat.set_yticklabels(muts, fontsize=1.7)
    ax_mat.tick_params(axis='y', length=1, width=.3, pad=.5)
    ax_mat.set_xlabel('Cells', labelpad=1)
    for s in ax_mat.spines.values():
        s.set_linewidth(.5)

    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ax_cb)
    cb.set_label('Allelic Frequency', size=6, labelpad=2)
    cb.ax.tick_params(labelsize=5, length=1.5, width=.4)
    cb.outline.set_linewidth(.4)
    cb.solids.set_rasterized(True)       # a smooth gradient: a raster cannot show an artefact
    cb.solids.set_edgecolor('none')
    cb.solids.set_linewidth(0)

    # same horizontal placement for tree and matrix (plot_tree sets an equal aspect), shorter colorbar at the matrix's top right
    pos, p = ax_mat.get_position(), ax_tree.get_position()
    ax_tree.set_position([pos.x0, p.y0, pos.width, p.height])
    cbh = .42 * pos.height
    ax_cb.set_position([pos.x1 + .018, pos.y1 - cbh, .022, cbh])
    fig.canvas.draw()
    t = [ax_tree.transData.transform((v, 0))[0] for v in (lx.values[0], lx.values[-1])]
    h = [ax_mat.transData.transform((v, 0))[0] for v in (lx.values[0], lx.values[-1])]
    assert np.allclose(t, h, atol=.5), 'tree and matrix columns are not aligned'

    fig.savefig(FIGURES / 'tree_heatmap.pdf', dpi=1000)       # vector; dpi only affects the rasterized colorbar
    for dpi, name in ((600, 'tree_heatmap.png'), (1000, 'tree_heatmap_1000dpi.png'), (2400, 'tree_heatmap_2400dpi.png')):
        fig.savefig(FIGURES / name, dpi=dpi)
    plt.close(fig)


def clone_tree(afm, tree):
    n_clones = afm.obs['MiTo_clone'].astype(str).replace('unassigned', np.nan).nunique()
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    mt.pl.plot_clone_tree(tree, ax=ax, linewidth=.4, max_radius=.075)
    ax.set_aspect('equal')
    plu.format_ax(ax, title=f'{n_clones} clones (circle area = cells)')
    for t in ax.texts:
        t.set_fontsize(6)
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.93)
    fig.savefig(FIGURES / 'clone_tree.pdf')
    fig.savefig(FIGURES / 'clone_tree.png', dpi=600)
    plt.close(fig)


if __name__ == '__main__':
    make_dirs()
    afm, tree = final_afm_and_tree()
    tree_heatmap(afm, tree)
    clone_tree(afm, tree)
