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
def triaged_adjusted_rand_score(y_true, y_pred):
    # assign each triaged point its own label
    k = y_pred.max() + 1
    invalid = np.flatnonzero(y_pred < 0)
    y_pred2 = y_pred.copy()
    y_pred2[invalid] = k + np.arange(invalid.size)
    
    return metrics.adjusted_rand_score(y, y_pred2)


# %% [markdown]
# # scikit-learn blob datasets

# %%
n_samples = 500
seed = 14
X_blob, y_blob = datasets.make_blobs(
    centers=np.array([[0, 0], [5, 0], [5, 4]]),
    n_samples=n_samples,
    random_state=seed,
)

random_state = 170
X, y_skew = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_skew = np.dot(X, transformation)

X_varied, y_varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

blob_datasets = dict(blob=(X_blob, y_blob), skew=(X_skew, y_skew), varied=(X_varied, y_varied))

# %%
blob_clusterers = {name: HDBSCAN().fit(X) for name, (X, y) in blob_datasets.items()}

# %%
fig, axes = plt.subplots(nrows=2, ncols=3, layout="constrained", sharex="col", gridspec_kw=dict(wspace=0.1))
for column, (name, (X, y)) in zip(axes.T, blob_datasets.items()):
    true_scatter(column[0], *X.T, y)
    pred_scatter(column[1], *X.T, blob_clusterers[name].labels_)
    
    ari = triaged_adjusted_rand_score(y, blob_clusterers[name].labels_)
    
    column[0].set_title(name)
    column[1].set_title(f"HDBSCAN: triaged ARI = {ari:0.2f}")

# %% tags=[]
methods = {
    "Meet": consensus.MeetConsensus,
    "LCA": consensus.LatentClassConsensus,
    "SBM": consensus.SBMConsensus,
    "IndEdges": consensus.BernoulliEdgeConsensus,
    "Gaussian": consensus.GaussianOneHotConsensus,
}

# %% tags=[]
rg = np.random.default_rng(5)

blob_bags = {}

for name, (X, y) in blob_datasets.items():
    if name not in blob_bags:
        blob_bags[name] = {}
    print(f"{blob_bags[name].keys()=}")
    
    for method, cls in methods.items():
        print(f"{name=} {method=}")
        if name in blob_bags and method in blob_bags[name]:
            print("done, skip")
            continue
        
        bag = consensus.BootstrapHDBSCAN(random_state=rg)
        bag_labels = bag.fit_transform(X)
        
        blob_bags[name][method] = bag_labels

# %%
fig, axes = plt.subplots(ncols=1 + 5, nrows=len(blob_datasets), sharey="row", layout="constrained")
for i, (row, (name, (X, y))) in enumerate(zip(axes, blob_datasets.items())):
    true_scatter(row[0], *X.T, y)
    row[0].set_title("true")
    
    for method, y_ in blob_bags[name].items():
        for s, (ax, pred) in enumerate(zip(row[1:], y_.T)):
            ari = triaged_adjusted_rand_score(y, pred)
            ax.set_title(f"boot $s={s}$ ARI={ari:0.2f}")
            pred_scatter(ax, *X.T, pred)

# %%
blob_pipelines = {}
blob_consensus = {}
rg = np.random.default_rng(5)

# %%
1

# %% tags=[]
for name, (X, y) in blob_datasets.items():
    if name not in blob_pipelines:
        blob_pipelines[name] = {}
    if name not in blob_consensus:
        blob_consensus[name] = {}
    print(f"{blob_consensus[name].keys()=}")
    
    for method, cls in methods.items():
        print(f"{name=} {method=}")
        if name in blob_consensus and method in blob_consensus[name]:
            print("done, skip")
            continue
        
        bag = consensus.BootstrapHDBSCAN(random_state=rg)
        kw = {}
        params = inspect.signature(cls).parameters
        needs_k = "n_blocks" in params
        if needs_k:
            kw["n_blocks"] = 3
        is_random = "random_state" in params
        if is_random:
            kw["random_state"] = rg
        
        cons = cls(**kw)
        
        bag_labels = bag.fit_transform(X)
        # print(f"{bag_labels=}")
        preds = cons.fit_predict(bag_labels)
        
        blob_pipelines[name][method] = (bag, cons)
        blob_consensus[name][method] = preds

# %%
Path("/Users/charlie/consensus/data/").mkdir(exist_ok=True)
with open("/Users/charlie/consensus/data/blob_data.pkl", "wb") as jar:
    cloudpickle.dump(
        dict(blob_pipelines=blob_pipelines, blob_consensus=blob_consensus),
        jar,
    )

# %%
fig, axes = plt.subplots(ncols=2 + len(blob_consensus["blob"]), nrows=len(blob_datasets), sharey="row", layout="constrained")
for i, (row, (name, (X, y))) in enumerate(zip(axes, blob_datasets.items())):
    true_scatter(row[0], *X.T, y)

    ari = triaged_adjusted_rand_score(y, blob_clusterers[name].labels_)
    row[0].set_title("true")
    row[1].set_title(f"HDBSCAN ARI={ari:0.2f}")
        
    pred_scatter(row[1], *X.T, blob_clusterers[name].labels_)
    
    if name in blob_consensus:
        for ax, (method, cons) in zip(row[2:], blob_consensus[name].items()):
            ari = triaged_adjusted_rand_score(y, cons)
            ax.set_title(f"{method} ARI={ari:0.2f}")
            # print(f"{name=} {method=}")
            # print(f"{np.unique(cons, return_counts=True)=}")
            pred_scatter(ax, *X.T, cons)

# %%
nmethods = sum(1 for (k, (bag, cons)) in next(iter(blob_pipelines.values())).items() if hasattr(cons, "objectives_"))
fig, axes = plt.subplots(ncols=nmethods, nrows=len(blob_datasets), squeeze=False, layout="constrained")
for i, (row, (name, (X, y))) in enumerate(zip(axes, blob_datasets.items())):
    the_methods = {k: cons for (k, (bag, cons)) in blob_pipelines[name].items() if hasattr(cons, "objectives_")}
    for ax, (method, cons) in zip(row, the_methods.items()):
        ax.set_title(f"{method}")
        ax.plot(-np.array(cons.objectives_)[10:])
        # print(cons.probabilities_)
        # ax.semilogy()
        print(f"{name=} {method=}")
        # print(f"{len(cons.objectives_)=} {np.array(cons.objectives_).shape=}")
        # print(f"{np.isnan(cons.objectives_).sum()=}")
        print(f"{(np.diff(cons.objectives_)>0).sum()=}")
        # print(f"{np.isinf(cons.objectives_).sum()=}")

# %%

# %%

# %%
nmethods = sum(1 for (k, (bag, cons)) in next(iter(blob_pipelines.values())).items() if hasattr(cons, "objectives_"))
fig, axes = plt.subplots(ncols=nmethods, nrows=len(blob_datasets), squeeze=False, layout="constrained")
for i, (row, (name, (X, y))) in enumerate(zip(axes, blob_datasets.items())):
    the_methods = {k: cons for (k, (bag, cons)) in blob_pipelines[name].items() if hasattr(cons, "objectives_")}
    for ax, (method, cons) in zip(row, the_methods.items()):
        ax.set_title(f"{method}")
        # ax.plot(-cons.objectives_[10:])
        print(cons.pi_)
        print(f"{cons.B_.shape=}")
        im = ax.imshow(cons.B_.reshape(-1, cons.B_.shape[2]).T, aspect="auto")
        ax.set_ylabel("K")
        plt.colorbar(im, ax=ax)
        # print(cons.probabilities_)
        # ax.semilogy()
        print(f"{name=} {method=}")
        # print(f"{len(cons.objectives_)=} {np.array(cons.objectives_).shape=}")
        # print(f"{np.isnan(cons.objectives_).sum()=}")
        # print(f"{(np.diff(cons.objectives_)>0).sum()=}")
        # print(f"{np.isinf(cons.objectives_).sum()=}")

# %%
nmethods = sum(1 for (k, (bag, cons)) in next(iter(blob_pipelines.values())).items() if hasattr(cons, "objectives_"))
fig, axes = plt.subplots(ncols=nmethods, nrows=len(blob_datasets), squeeze=False, layout="constrained")
for i, (row, (name, (X, y))) in enumerate(zip(axes, blob_datasets.items())):
    the_methods = {k: cons for (k, (bag, cons)) in blob_pipelines[name].items() if hasattr(cons, "objectives_")}
    for ax, (method, cons) in zip(row, the_methods.items()):
        ax.set_title(f"{method}")
        # ax.plot(-cons.objectives_[10:])
        print(cons.pi_)
        print(f"{cons.B_.shape=}")
        im = ax.imshow(cons.B_.transpose(0, 2, 1).reshape(-1, cons.B_.shape[1]).T, aspect="auto")
        ax.set_ylabel("T")
        plt.colorbar(im, ax=ax)
        # print(cons.probabilities_)
        # ax.semilogy()
        print(f"{name=} {method=}")
        # print(f"{len(cons.objectives_)=} {np.array(cons.objectives_).shape=}")
        # print(f"{np.isnan(cons.objectives_).sum()=}")
        # print(f"{(np.diff(cons.objectives_)>0).sum()=}")
        # print(f"{np.isinf(cons.objectives_).sum()=}")

# %%
nmethods = sum(1 for (k, (bag, cons)) in next(iter(blob_pipelines.values())).items() if hasattr(cons, "objectives_"))
fig, axes = plt.subplots(ncols=nmethods, nrows=len(blob_datasets), squeeze=False, layout="constrained")
for i, (row, (name, (X, y))) in enumerate(zip(axes, blob_datasets.items())):
    the_methods = {k: cons for (k, (bag, cons)) in blob_pipelines[name].items() if hasattr(cons, "objectives_")}
    for ax, (method, cons) in zip(row, the_methods.items()):
        ax.set_title(f"{method}")
        # ax.plot(-cons.objectives_[10:])
        print(cons.pi_)
        print(f"{np.unique(cons.probabilities_.argmax(1), return_counts=True)=}")
        im = ax.imshow(cons.probabilities_.T, aspect="auto")
        plt.colorbar(im, ax=ax)
        # print(cons.probabilities_)
        # ax.semilogy()
        print(f"{name=} {method=}")
        # print(f"{len(cons.objectives_)=} {np.array(cons.objectives_).shape=}")
        # print(f"{np.isnan(cons.objectives_).sum()=}")
        # print(f"{(np.diff(cons.objectives_)>0).sum()=}")
        # print(f"{np.isinf(cons.objectives_).sum()=}")

# %%

# %%
