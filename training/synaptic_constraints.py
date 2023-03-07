# Heterosynaptic competition functions 
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init 
from torch.nn import functional as F
from scipy.spatial.distance import cdist


# Heterosynaptic competition rule functions
relu = torch.nn.ReLU()
def hsc_thresh(weights, W_max): 
    theta_i = relu(torch.sum(weights, dim=0) - W_max)
    theta_j = relu(torch.sum(weights, dim=1) - W_max)
    return theta_i, theta_j

def hsc_update(model, ei_ratio, W_max_E, W_max_I, w_max, epsilon, eta):
    hidden_sgns = model.rnn.hidden_sgns
    e_neurons = (hidden_sgns == 1).nonzero().squeeze()
    i_neurons = (hidden_sgns == -1).nonzero().squeeze()
    
    W_current = torch.clone(model.rnn.h2h.weight.data)
    W_current[e_neurons,:] = torch.clamp(W_current[e_neurons,:], 0, float('Inf'))
    W_current[i_neurons,:] = torch.clamp(W_current[i_neurons,:], -float('Inf'), 0)

   # Applying threshold
    e_weights = W_current[e_neurons,:]
    i_weights = W_current[i_neurons,:]
    e_theta_pre, e_theta_post = hsc_thresh(e_weights, W_max_E)
    i_theta_pre, i_theta_post = hsc_thresh(torch.abs(i_weights), W_max_I)
    HSC_e_thresh_update = eta*epsilon*(e_theta_pre.repeat(e_weights.size(dim = 0), 1) + 
                                       e_theta_post.repeat(e_weights.size(dim = 1), 1).T)
    HSC_i_thresh_update = eta*epsilon*(i_theta_pre.repeat(i_weights.size(dim = 0), 1) + 
                                       i_theta_post.repeat(i_weights.size(dim = 1), 1).T)

    return torch.clamp(e_weights - HSC_e_thresh_update, min = 0, max = w_max), torch.clamp(i_weights + HSC_i_thresh_update, min = -w_max, max = 0)


# Long-range wiring penalty 

def sheet_dist(N_x, N_y):
    x1 = np.linspace(-(N_x)//2,(N_x)//2-1,N_x)
    x1 = np.expand_dims(x1,axis=1)
    x2 = np.linspace(-(N_y)//2,(N_y)//2-1,N_y)
    x2 = np.expand_dims(x2,axis=1)
    x_coordinates = np.expand_dims(np.repeat(x1,N_y,axis = 0).reshape(N_x,N_y).transpose().flatten(),axis=1)
    y_coordinates = np.expand_dims(np.repeat(x2,N_x,axis = 0).reshape(N_x,N_y).flatten(),axis=1)
    
    #calculate torus distance on 2d sheet
    distances_x = cdist(x_coordinates,x_coordinates)
    distances_y = cdist(y_coordinates,y_coordinates)
    
    distances = np.sqrt(np.square(distances_x) + np.square(distances_y))
    dist = distances.reshape(N_y,N_x,N_y,N_x)
    dist = dist.reshape(N_x*N_y,N_x*N_y)
    
    return dist

def dist_cost_update(model, N_x, N_y, d_thresh, delta):
    hidden_sgns = model.rnn.hidden_sgns 
    e_neurons = (hidden_sgns == 1).nonzero().squeeze()
    i_neurons = (hidden_sgns == -1).nonzero().squeeze()
    ds = sheet_dist(N_x,N_y)
    W_current = torch.clone(model.rnn.h2h.weight.data)
    
    e_weights = W_current[e_neurons,:]
    ds_e_far = ds[e_neurons, :]
    #ds_e_far[ds[e_neurons,:] > d_thresh] = 1 
    ds_e_far[ds[e_neurons,:] <= d_thresh] = 0
    W_current[e_neurons, :] = relu(e_weights - delta*(1/ds_e_far.max())*ds_e_far)

    i_weights = W_current[i_neurons,:]
    ds_i_far = ds[i_neurons, :]
    #ds_i_far[ds[i_neurons,:] > d_thresh] = 1 
    ds_i_far[ds[i_neurons,:] <= d_thresh] = 0
    W_current[i_neurons, :] = -relu(-i_weights - delta*(1/ds_i_far.max())*ds_i_far)
    
    return W_current[e_neurons, :], W_current[i_neurons, :]