import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP

class MLPNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # PyG's MLP class automatically handles layers and dropout
        self.mlp = MLP([in_channels, hidden_channels, out_channels], dropout=0.5)

    def forward(self, x, edge_index=None):
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)