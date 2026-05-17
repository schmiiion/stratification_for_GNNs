import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # heads=8, out_channels=8 => 64 total hidden units
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.5)
        self.conv2 = GATConv(64, out_channels, heads=1, concat=False, dropout=0.5)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)