import numpy as np
import torch
import torch.nn as nn
from torch.nn import init 
from torch.nn import functional as F 
from training.analysis import get_neuron_idxs_in_big_module

torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RNN functions 
class EI_CTRNN(nn.Module):
    """Continuous-time RNN.
    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        mask: N_h x N_h mask either 2d or 1d
    Inputs:
        input: (seq_len, batch, input_size), network input
        hidden: (batch, hidden_size), initial hidden activity
    """

    def __init__(self, input_size, hidden_size, ei_ratio, P=None, dt=None, sigma_rec=0, mask = None, **kwargs):

        super().__init__()
        self.ei_ratio = ei_ratio
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        self.mask = mask        
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha
        self._sigma = np.sqrt(2 / alpha) * sigma_rec
        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        
        if P is not None:
            self.P = P.detach().cpu().numpy().astype(int)
        else: 
            self.P = None

        #self.reset_parameters()
        
        #initialize hidden to hidden weight matrix using the mask
        if mask is None:
            temp = 0
        else:
            self.h2h.weight.data = self.h2h.weight.data*torch.nn.Parameter(mask)
        
        self.hidden_sgns = 2*torch.bernoulli(ei_ratio*torch.ones(hidden_size))-1
        self.h2h.weight.data = (torch.abs(self.h2h.weight.data).T*self.hidden_sgns).T
        self.e_neurons = (self.hidden_sgns == 1).nonzero().squeeze()
        self.i_neurons = (self.hidden_sgns == -1).nonzero().squeeze()
        

    def reset_parameters(self):
        nn.init.eye_(self.h2h.weight)
        self.h2h.weight.data *= 0.5

    def init_hidden(self, input):
        batch_size = input.shape[1]
        return torch.zeros(batch_size, self.hidden_size).to(input.device)

    def recurrence(self, input, hidden):
        """Recurrence helper."""
        pre_activation = self.input2h(input) + self.h2h(hidden)
        # recurrent unit noise 
        mean = torch.zeros_like(pre_activation)
        std = self._sigma
        noise_rec = torch.normal(mean=mean, std=std)
        
        # Add noise only to neurons corresponding to biggest module  
        if self.P is not None: 
          cluster_assignments = self.P # this will be a numpy array
          big_mod_idxs = get_neuron_idxs_in_big_module(cluster_assignments) 
          big_mod_idxs = torch.from_numpy(big_mod_idxs) # make it a tensor
          noise_mask = torch.zeros_like(noise_rec)
          noise_mask[:, big_mod_idxs] = 1 
          noise_rec *= noise_mask 


        pre_activation += noise_rec
        # self.pre_activation = pre_activation
        h_new = torch.relu(hidden * self.oneminusalpha +
                           pre_activation * self.alpha)
        
        return h_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(input)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)
        output = torch.stack(output, dim=0)
        return output, hidden

class RNNNet(nn.Module):
    
    """Recurrent network model.
    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """

    def __init__(self, input_size, hidden_size, output_size, mask, **kwargs):
        super().__init__()

        #Continuous time RNN
        self.rnn = EI_CTRNN(input_size, hidden_size, mask = mask, **kwargs)
        #readout layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #hidden dynamics
        rnn_activity, _ = self.rnn(x)
        #readout
        out = self.fc(rnn_activity)
        return out, rnn_activity