# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python [conda env:b]
#     language: python
#     name: conda-env-b-py
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import torch
from sklearn import datasets, metrics
from sklearn.pipeline import Pipeline
import consensus
from hdbscan import HDBSCAN
import inspect
import pickle
from pathlib import Path
import cloudpickle
from torchvision.datasets import MNIST
from sklearn.decomposition import PCA, NMF, DictionaryLearning
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS, SpectralEmbedding

# %%
# %config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import offset_copy
from matplotlib.patches import Ellipse, Rectangle, ConnectionPatch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import contextlib
import colorcet as cc

plt.rc("figure", dpi=300)
plt.rc("figure", figsize=(7, 4))
SMALL_SIZE = 6.5
MEDIUM_SIZE = 8
BIGGER_SIZE =  10
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "serif",
})
preamble = r"""
\renewcommand{\familydefault}{\sfdefault}
\usepackage[scaled=1]{helvet}
\usepackage[helvet]{sfmath}
\usepackage{textgreek}
"""
plt.rc('text.latex', preamble=preamble)
plt.rc('svg', fonttype='none')
plt.rc('ps', usedistiller='xpdf')
plt.rc('pdf', fonttype=42)

def clearpanel(figure):
    figure.set_facecolor([0, 0, 0, 0])
    figure.patch.set_facecolor([0, 0, 0, 0])


# %%
def true_scatter(ax, x1, x2, y):
    return ax.scatter(x1, x2, c=y, cmap=cc.m_glasbey_cool, lw=0, s=3)

def pred_scatter(ax, x1, x2, y):
    valid = y >= 0
    s1 = ax.scatter(x1[~valid], x2[~valid], c="k", lw=0, s=3)
    s2 = ax.scatter(x1[valid], x2[valid], c=y[valid], cmap=cc.m_glasbey_light, lw=0, s=3)
    return s1, s2


# %%
def embedding_scatter(fig, Xe, pred, true=True, title="true labels", T=False, ylabel=True, xlabel=True):
    ncols = Xe.shape[1] - 1
    nrows = 1
    sharey = True
    sharex = False
    if T:
        ncols = 1
        nrows = Xe.shape[1] - 1
        sharex = True
        sharey = False
    axes = subfig.subplots(ncols=ncols, nrows=nrows, sharey=sharey, sharex=sharex)
    for j, ax in enumerate(axes):
        if true:
            if T:
                true_scatter(ax, Xe[:, 0], Xe[:, j + 1], pred)
            else:
                true_scatter(ax, Xe[:, j + 1], Xe[:, 0], pred)
        else:
            if T:
                pred_scatter(ax, Xe[:, 0], Xe[:, j + 1], pred)
            else:
                pred_scatter(ax, Xe[:, j + 1], Xe[:, 0], pred)
        if T:
            if ylabel:
                ax.set_ylabel(f"tsne{j+1}")
            if xlabel and (j == nrows - 1):
                ax.set_xlabel("tsne0")
        else:
            if ylabel and (j == 0):
                ax.set_ylabel("tsne0")
            if xlabel:
                ax.set_xlabel(f"tsne{j+1}")
        if not ylabel:
            ax.set_yticklabels([])
        if not xlabel:
            ax.set_xticklabels([])
    if T:
        axes.flat[0].set_title(title)
    else:
        fig.suptitle(title)
        


# %%
def triaged_adjusted_rand_score(y_true, y_pred):
    # assign each triaged point its own label
    k = y_pred.max() + 1
    invalid = np.flatnonzero(y_pred < 0)
    y_pred2 = y_pred.copy()
    y_pred2[invalid] = k + np.arange(invalid.size)
    
    return metrics.adjusted_rand_score(y, y_pred2)


# %% [markdown]
# mnist embedding experiments

# %%
mnist_train = MNIST("/Users/charlie/data/MNIST", download=True)

# %%
n_datapoints = 2500
rg = np.random.default_rng(0)
which = rg.choice(len(mnist_train), size=n_datapoints, replace=False)
which.sort()
X, y = mnist_train.data[which], mnist_train.targets[which]
X.shape, y.shape, X.dtype, y.dtype

# %%
X = X.reshape(n_datapoints, 784).numpy(force=True).astype(float)

# %%
rank = 5
# Xe = PCA(rank).fit_transform(X)
# Xe = LocallyLinearEmbedding(n_components=rank).fit_transform(X)
Xe = TSNE(n_components=3, random_state=0).fit_transform(X)
# Xe = MDS(n_components=rank, random_state=0, normalized_stress="auto").fit_transform(X)
# Xe = SpectralEmbedding(n_components=rank, random_state=0).fit_transform(X)
# Xe = NMF(n_components=rank, random_state=0).fit_transform(X)
# Xe = DictionaryLearning(n_components=rank, random_state=0).fit_transform(X)

# %%
hdbscan = HDBSCAN()
hdbscan.fit(Xe)

# %%
fig = plt.figure(layout="constrained")
figs = fig.subfigures(ncols=2)

for i, subfig in enumerate(figs):
    y_ = y
    title = "true labels"
    if i:
        y_ = hdbscan.labels_
        ari = triaged_adjusted_rand_score(y, y_)
        title = f"HDBSCAN ARI={ari:0.2f}"
    embedding_scatter(subfig, Xe, y_, true=not i, title=title, T=True)

# %%
nbags = 10

rg = np.random.default_rng(0)
bag = consensus.BootstrapHDBSCAN(random_state=rg, n_subsets=nbags, subset_fraction=0.9)
bag_labels = bag.fit_transform(X)

# %%
fig = plt.figure(layout="constrained")
figs = fig.subfigures(ncols=1 + min(5, nbags))

for i, subfig in enumerate(figs):
    y_ = y
    title = "true labels"
    if i:
        y_ = bag_labels[:, i - 1]
        ari = triaged_adjusted_rand_score(y, y_)
        title = f"$s={i-1}$ ARI={ari:0.2f}"
    embedding_scatter(subfig, Xe, y_, true=not i, title=title, T=1, ylabel=i == 0)

# %% tags=[]
methods = {
    "Meet": consensus.MeetConsensus,
    "LCA": consensus.LatentClassConsensus,
    "SBM": consensus.SBMConsensus,
    "IndEdges": consensus.BernoulliEdgeConsensus,
    # "Gaussian": consensus.GaussianOneHotConsensus,
}

# %%
blob_pipelines = {}
blob_consensus = {}
rg = np.random.default_rng(5)

# %%
for method, cls in methods.items():
    print(f"{method=}")
    if method in blob_consensus:
        print("done, skip")
        continue

    bag = consensus.BootstrapHDBSCAN(random_state=rg, subset_fraction=0.9)
    kw = {}
    params = inspect.signature(cls).parameters
    needs_k = "n_blocks" in params
    if needs_k:
        kw["n_blocks"] = 10
    is_random = "random_state" in params
    if is_random:
        kw["random_state"] = rg

    cons = cls(**kw)

    bag_labels = bag.fit_transform(X)
    # print(f"{bag_labels=}")
    preds = cons.fit_predict(bag_labels)

    blob_pipelines[method] = (bag, cons)
    blob_consensus[method] = preds

# %%
