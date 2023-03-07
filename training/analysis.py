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
import matplotlib.pyplot as plt
import copy

def get_act_act_variances(activity_list_all_tasks, len_tasks, N_x, N_y):
  #activity_list_all_tasks will be num_tasks x num_trials x len_trial x 1 x num_nodes. len_trials can be variable
    activity_list_all_combined = [item for sublist in activity_list_all_tasks for item in sublist]
    len_activities = list(map(len,activity_list_all_combined))
    min_length = min(len_activities)
    max_length = max(len_activities)
    len_activities_values = np.bincount(len_activities).nonzero()[0]
    #len_activities_values = [len_activities_values[0]]  #### If want to use only shortest length; also set num_
    num_len_activities_values = len_activities_values.shape[0]
    activity_var_corr = np.zeros((num_len_activities_values,N_x*N_y,N_x*N_y))
    num_at_length = np.zeros(num_len_activities_values)
    for idx, iter_len in enumerate(len_activities_values):
        which_activites = np.where(np.array(len_activities)==iter_len)[0]
        num_at_length[idx] = which_activites.shape[0]
        activities_of_fixed_len = [activity_list_all_combined[x] for x in which_activites]
        activities_of_fixed_len = np.concatenate(activities_of_fixed_len,axis=1) #shape task_len,num_trials,N
        activity_variance = np.var(activities_of_fixed_len, axis=1)
        act_var_cor_idx = np.corrcoef(activity_variance.T)
        act_var_cor_idx[np.isnan(act_var_cor_idx)]=0
        activity_var_corr[idx] = act_var_cor_idx

    activity_var_corr_averaged = np.average(activity_var_corr,axis=0,weights=num_at_length)

    return activity_var_corr_averaged

def get_activity(net, env, device, num_trial=1000):

    """Get activity of equal-length trials"""

    trial_list = list()
    activity_list = list()
    perf=0
    for i in range(num_trial):
        env.new_trial()
        ob, gt = env.ob, env.gt
        ob = ob[:, np.newaxis, :]  # Add batch axis
        inputs = torch.from_numpy(ob).type(torch.DoubleTensor).to(device)
        action_pred, activity = net(inputs)
        activity = activity.detach().cpu().numpy()
        action_pred = action_pred.detach().cpu().numpy()
        action_pred = np.argmax(action_pred, axis=-1)
        trial_list.append(env.trial)
        activity_list.append(activity)
        perf += gt[-1] == action_pred[-1, 0]

    perf = perf*1./num_trial
    return activity_list, trial_list, perf

def get_performance(net, env, num_trial=1000, device='cpu'):
    perf = 0

    for i in range(num_trial):

        env.new_trial()
        ob, gt = env.ob, env.gt
        ob = ob[:, np.newaxis, :]  # Add batch axis
        inputs = torch.from_numpy(ob).type(torch.DoubleTensor).to(device)
        action_pred, _ = net(inputs)
        action_pred = action_pred.detach().cpu().numpy()
        action_pred = np.argmax(action_pred, axis=-1)
        perf += gt[-1] == action_pred[-1, 0]

    perf /= num_trial
    return perf

def get_activity_list(net,env,tasks,device,num_trial=500):
    task_variance_list = list()
    activity_list_all_tasks = [None]*len(tasks)
    perf_per_task = np.zeros(len(tasks))

    for i in range(len(tasks)):
        env.set_i(i)
        #Compute activity for the given trial
        # num_trial = 500
        activity_list, trial_list,perf = get_activity(net, env, device, num_trial=num_trial)
        activity_list_all_tasks[i] = activity_list
        perf_per_task[i] = perf
    return activity_list_all_tasks,perf_per_task


# Activity functions with accessibility to within-period activity
def activity_list(net, tasks, env, device, num_trial=100):
    """
    Get network activity during each task over a given number of trials. 
    Returns:
        - activity_list: (num_trial x num_tasks x time x batch x num_neurons) 
            - e.g. activity_list[i]['go'][:,:,:j] gets activity of neurons 0,1,...,j-1 in the i-th trial 
            of go() task across the full duration of the task. 
        - activity_per_period_list: (num_trial x num_tasks x num_periods x time x batch x num_neurons)
            - e.g. activity_per_period_list[i]['go']['stimulus'][:,:,:j] gets activity of neurons 0,1,...,j-1
            in the i-th trial of the go() task during the 'stimulus' period. 
    """
    num_tasks = len(tasks)
    activity_list = list()
    activity_per_period_list = list()
    for j in range(num_trial): 
        # get activity in each task for a single trial
        activity_dict_single_trial = {} # these dictionaries allow accessing activity by task name
        activity_per_period_dict_single_trial = {}
        for i in range(num_tasks):
            env.new_trial()    
            # print(env.task.task_name)
            # print(env.task.timing)
            ob, gt = env.ob, env.gt
            ob = ob[:, np.newaxis, :]
            inputs = torch.from_numpy(ob).type(torch.DoubleTensor).to(device)
            action_pred, activity = net(inputs)
            activity = activity.detach().cpu().numpy()
            # plt.plot(ob.squeeze(axis=1))
            # plt.plot(action_pred)
            # print(f'Ground truth: {gt[-1]}, Predicted: {action_pred[-1, 0]}')
            # plt.show()
            
            # Per-period activity
            period_idx = {} # activity[period_idx['period']] will slice activity array to give activity in 'period'
            running_idx = 0
            for period in env.task.timing.keys(): # loop gives time indices corresponding to 'period'
                period_idx[period] = np.arange(running_idx, running_idx + int(env.task.timing[period] // env.dt))
                running_idx += int(env.task.timing[period] // env.dt)
            activity_dict_single_trial[env.task.task_name] = activity 
            activity_per_period = {}
            for period in period_idx:
                activity_per_period[period] = activity[period_idx[period], :, :]
            activity_per_period_dict_single_trial[env.task.task_name] = activity_per_period
            
        activity_list.append(activity_dict_single_trial)
        activity_per_period_list.append(activity_per_period_dict_single_trial)

    return activity_list, activity_per_period_list

def activity_by_module(net, tasks, env, device, P, num_trial=100):
    """
    Get network activity in each module.
    Args: 
        - tasks: list of tasks 
        - env: environment object, * must be scheduled with SequentialSchedule * 
        - P: cluster assignments
    Returns: 
        - activity_list_by_module: ( (num_modules, neuron_idxs) x num_trial x num_tasks x time x batch x num_neurons in module) 
            - e.g. activity_list_by_module[m][0][i]['go'][:,:,:] gets activity of the neurons in the m-th module 
            in the i-th trial of go() task across its full duration. 
        - activity_per_period_list_by_module: ((num_modules, neuron_idxs), x num_tasks x num_periods x time x batch x num_neurons)
            - e.g. activity_per_period_list_by_module[m][i]['go']['stimulus'][:,:,:] gets activity of neurons in the m-th module
            in the i-th trial of the go() task during the 'stimulus' period. 
            
        - Note that each item in the list is a tuple where the first element is the activity, and the second element
        are the indices corresponding to the neurons in that module. 
    """
    if type(P) == torch.Tensor:
        P = P.detach().cpu().numpy()
        
    P = P[:, np.any(P, axis=0)]
    num_modules = P.shape[1] 
    tmp_activity_list, activity_per_period_list, = activity_list(net, tasks, env, device, num_trial=num_trial)
    activity_list_by_module = list() 
    activity_per_period_list_by_module = list() 
    for j in range(num_modules):
        module = P[:, j]
        neuron_idxs = np.nonzero(module)[0]
        module_activity_list = copy.deepcopy(tmp_activity_list)
        module_activity_per_period_list = copy.deepcopy(activity_per_period_list) 

        for i in range(len(module_activity_list)): 

            for task_name in list(module_activity_list[i].keys()): 
                module_activity_list[i][task_name] = tmp_activity_list[i][task_name][:, :, neuron_idxs]

                for period in list(module_activity_per_period_list[i][task_name].keys()):
                    module_activity_per_period_list[i][task_name][period] = activity_per_period_list[i][task_name][period][:, :, neuron_idxs]

        activity_list_by_module.append((module_activity_list, neuron_idxs))
        activity_per_period_list_by_module.append((module_activity_per_period_list, neuron_idxs))
    
    return activity_list_by_module, activity_per_period_list_by_module 