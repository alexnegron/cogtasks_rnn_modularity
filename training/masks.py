from scipy import sparse
from scipy.spatial.distance import cdist
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init 
from torch.nn import functional as F 

def mask2d(N_x,N_y,cutoff,periodic):
    x1 = np.linspace(-(N_x)//2,(N_x)//2-1,N_x)
    x1 = np.expand_dims(x1,axis=1)
    x2 = np.linspace(-(N_y)//2,(N_y)//2-1,N_y)
    x2 = np.expand_dims(x2,axis=1)
    x_coordinates = np.expand_dims(np.repeat(x1,N_y,axis = 0).reshape(N_x,N_y).transpose().flatten(),axis=1)
    y_coordinates = np.expand_dims(np.repeat(x2,N_x,axis = 0).reshape(N_x,N_y).flatten(),axis=1)
    
    #calculate torus distance on 2d sheet
    distances_x = cdist(x_coordinates,x_coordinates)
    distances_y = cdist(y_coordinates,y_coordinates)
    
    if(periodic==True):
        distances_y = np.minimum(N_y-distances_y,distances_y)
        distances_x = np.minimum(N_x-distances_x,distances_x)
    
    distances = np.sqrt(np.square(distances_x) + np.square(distances_y))
    dist = distances.reshape(N_y,N_x,N_y,N_x)
    dist = dist.reshape(N_x*N_y,N_x*N_y)
    ori_dists = np.copy(dist)
    dist[ori_dists<=cutoff] = 1
    dist[ori_dists>cutoff] = 0
    return dist

def sparsemask2d(N_x,N_y,sparsity):
    elements = np.random.uniform(0,1,(N_x,N_y))
    mask = (elements<sparsity).astype(int)
    return mask
    
def smallworldmask(N_x,N_y,cutoff_local,cutoff_large, periodic):
    x1 = np.linspace(-(N_x)//2,(N_x)//2-1,N_x)
    x1 = np.expand_dims(x1,axis=1)

    x2 = np.linspace(-(N_y)//2,(N_y)//2-1,N_y)
    x2 = np.expand_dims(x2,axis=1)

    x_coordinates = np.expand_dims(np.repeat(x1,N_y,axis = 0).reshape(N_x,N_y).transpose().flatten(),axis=1)
    y_coordinates = np.expand_dims(np.repeat(x2,N_x,axis = 0).reshape(N_x,N_y).flatten(),axis=1)

    #calculate torus distance on 2d sheet
    distances_x = cdist(x_coordinates,x_coordinates)
    distances_y = cdist(y_coordinates,y_coordinates)

    if(periodic==True):
        distances_y = np.minimum(N_y-distances_y,distances_y)
        distances_x = np.minimum(N_x-distances_x,distances_x)
  
    distances = np.sqrt(np.square(distances_x) + np.square(distances_y))
    dist = distances.reshape(N_y,N_x,N_y,N_x)
    dist = dist.reshape(N_x*N_y,N_x*N_y)
    ori_dist = np.copy(dist)
  
    local = np.zeros_like(dist)
    local[dist<cutoff_local] = 1
    mask = np.copy(local)

    elements = np.zeros((N_x*N_y,N_x*N_y))
    for i in range(N_x*N_y):
        num_nonlocal = np.count_nonzero((ori_dist[i]>=cutoff_local)*(ori_dist[i]<=cutoff_large))
        # num_nonlocal = np.count_nonzero((ori_dist[i]<cutoff_large)) - np.count_nonzero(dist[i])
        placed=0
        while placed<num_nonlocal:
            rnd_idx = np.random.randint(0,N_x*N_y)
            if mask[i,rnd_idx]==1:
                continue
            else:
                mask[i,rnd_idx]=1
                placed=placed+1
      
    return mask