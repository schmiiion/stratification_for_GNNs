import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter, Linear
from torch_geometric.nn import MessagePassing, APPNP


class GPRGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, cfg):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

        # Pull GPR-specific args from your hydra config (with safe defaults)
        K = getattr(cfg, 'K', 10)
        alpha = getattr(cfg, 'alpha', 0.1)
        Init = getattr(cfg, 'Init', 'PPR')
        Gamma = getattr(cfg, 'Gamma', None)
        ppnp = getattr(cfg, 'ppnp', 'GPR_prop')
        self.dprate = getattr(cfg, 'dprate', 0.5)
        self.dropout = getattr(cfg, 'dropout', 0.5)

        if ppnp == 'PPNP':
            self.prop1 = APPNP(K, alpha)
        elif ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        # Allow standard forwarding or passing x, edge_index directly
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)