import numpy as np
import torch
import torch.nn as nn
from torch.nn import init 
from torch.nn import functional as F 
import math
import argparse
import random
import os
import time
import gym
import neurogym as ngym
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

from training.analysis import get_activity, get_performance, get_activity_list 

from collections import deque
ADJACENCY_EPS = 1e-15
def girvan_newman(adj:torch.Tensor, clusters:torch.Tensor):
    """Compute GN modularity statistic on adjacency matrix A using cluster assignments P.
    This version follows the equations in the GN paper fairly literally. The first line computes the cluster-cluster
    connectivity matrix 'e', which is then summed along rows or columns to get expected connectivity.
    """
    e = (clusters.T @ adj @ clusters) / torch.sum(adj)
    e_sum_row = torch.sum(e, dim=0)
    e_sum_col = torch.sum(e, dim=1)
    return torch.trace(e) - torch.sum(e_sum_col * e_sum_row)

def spectral_modularity(adj, max_clusters=None) -> torch.Tensor:
    """Spectral algorithm to quickly find an approximate maximum for the Girvan-Newman modularity score.
    See Newman, M. E. J. (2006). Modularity and community structure in networks. PNAS, 103(23), 8577–8582.
    """
    n = adj.size(0)
    max_clusters = max_clusters or n

    # Quit early if the adjacency matrix is all zeros (otherwise SVD below will error): return all-zeros for clustering
    if adj.mean() < ADJACENCY_EPS:
        return adj.new_zeros(n, max_clusters)

    # Normalize by sum of all 'edges' / AKA convert to probability assuming all A>=0
    adj = adj / adj.sum()
    # Compute degree of each 'vertex' / AKA compute marginal probability
    A1 = adj.sum(dim=1, keepdims=True)
    is_dead = A1.flatten() < ADJACENCY_EPS
    # Compute modularity matrix 'B': connectivity minus expected random connectivity
    B = adj - A1 * A1.T

    def gn_score(clusters):
        # Local helper function to compute Girvan-Newman modularity score using intermediate matrix 'B'
        return torch.sum(B * (clusters @ clusters.T))

    # Initialize everything into a single cluster, pruning out 'dead' units right away
    clusters = adj.new_zeros(n, max_clusters)
    clusters[~is_dead, 0] = 1.
    best_score = gn_score(clusters)
    # Iteratively subdivide until modularity score is not improved. This is a greedy method that takes any high-level
    # split that improves total modularity score. Each split is a branch of a binary tree, so after 'l' splits there
    # may be as many as 2^l modules, but some branches may be pruned early. This 'tree' is traversed breadth-first.
    # Tree traversal simply means keeping track of which column to try splitting next, which we keep track of in a
    # FIFO queue.
    queue, next_avail_col = deque([0]), 1
    while len(queue) > 0 and next_avail_col < max_clusters:
        col = queue.popleft()
        # Which variables are we working with here
        mask = clusters[:, col] == 1.
        # Skip this division if only one element left
        if mask.sum() <= 1:
            continue
        # Isolate the submatrix of B just containing these variables
        Bsub = B[mask, :][:, mask]
        # Compute single top eigenvectors for this sub-matrix
        _, _, v = torch.svd(Bsub)
        # Skip this division if the leading eigenvector does not contain both (+) and (-) elements. Otherwise, propose a
        # split based on the first eigenvector
        if torch.all(v[:, 0] >= 0.) or torch.all(v[:, 0] <= 0.):
            continue
        # Try new subdivision out: use current 'col' for (+) side of v, and use 'next_avail_col' for (-) side. In rare
        # cases, v will have exact zeros on nodes that are not 'dead' according to the EPS check above. The cluster
        # assignments for such nodes does not affect the score anyway (|v| is what matters), so we use >= in the (+)
        # cluster to include the zeros there arbitrarily.
        clusters[mask, col] = (v[:, 0] >= 0).float()
        clusters[mask, next_avail_col] = (v[:, 0] < 0).float()
        subdiv_score = gn_score(clusters)
        # If improved, keep it and push potential further subdivisions onto the queue
        if subdiv_score > best_score:
            queue.extend([col, next_avail_col])
            next_avail_col += 1
            best_score = subdiv_score
        # If not improved, undo change to 'clusters'
        else:
            clusters[:, col] = mask.float()
            clusters[:, next_avail_col] = 0.

    # Sanity check: only those units initially pruned as 'dead' should be missing
    assert torch.all((clusters.sum(dim=1) == 0.) == is_dead)

    return clusters

def sort_by_cluster(cluster, remove_dead=False):
    isort = torch.tensor(cluster).argmax(dim=1).argsort()
    if remove_dead:
        isort = torch.tensor([idx for idx in isort if cluster[idx, :].sum() != 0.])
    return isort

def merge_dicts(a, b):
    # Just wrapping this fancy-looking expression (Python 3.5+ only). https://stackoverflow.com/a/26853961
    return {**a, **b}

def plot_assoc(A, ax=None, lines_at=None, line_args={}, colorbar=False, vmin=0., vmax=None, cmap=None):
    ax = ax or plt.gca()
    mat = ax.imshow(A, cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(mat, cax=cax)
    
    if lines_at is not None:
        sz = len(A)
        line_args = merge_dicts({'color': 'r', 'linewidth':0.5}, line_args)
        for x in np.cumsum(lines_at)-0.5:
            ax.plot([-.5, sz-.5], [x, x], **line_args)
            ax.plot([x, x], [-.5, sz-.5], **line_args)
    # ax.set_xticks([])
    # ax.set_yticks([])


def spectral_modularity_maximization(model): 
    w = torch.abs(model.rnn.h2h.weight.data)
    wT = torch.transpose(w, 0, 1) 
    wTw = (wT@w).to(torch.float32)
    clusters_p = spectral_modularity(wTw).type(torch.float32)
    Q_score = girvan_newman(wTw, clusters=clusters_p)
    isrt = sort_by_cluster(clusters_p, remove_dead=True)
    return Q_score, isrt, clusters_p, wTw 