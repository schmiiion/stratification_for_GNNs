import torch
import torch.nn.functional as F
from torch_geometric.nn import MixHopConv

class MixHopNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # powers=[0, 1, 2] means it looks at the node itself, 1-hop, and 2-hop neighbors.
        # hidden_channels=64 means output is 64*3 = 192.
        self.conv1 = MixHopConv(in_channels, hidden_channels, powers=[0, 1, 2])
        self.conv2 = MixHopConv(hidden_channels * 3, out_channels, powers=[0, 1, 2])

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)