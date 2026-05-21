import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter, Linear
from torch_geometric.nn import MessagePassing, APPNP


class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = 10 #number of hops -> 10 (default)
        K= 10
        alpha = 0.1 #teleporting probability. -> 0.1 (default)

        # Init is PPR-like
        TEMP = alpha*(1-alpha)**np.arange(K+1)
        TEMP[-1] = (1-alpha)**K

        self.temp = Parameter(torch.tensor(TEMP))

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.dropout = 0.5
        self.dprate = 0.5

        self.prop1 = GPR_prop()


    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, adj_dict):
        edge_index = adj_dict['edge_idx']

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)