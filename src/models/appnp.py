import torch
import torch.nn.functional as F
from torch_geometric.nn import APPNP

class APPNPNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        # K is the number of propagation steps, alpha is teleport probability
        self.prop = APPNP(K=10, alpha=0.1)

    def forward(self, x, adj_dict):
        edge_index = adj_dict["edge_idx"]
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)
