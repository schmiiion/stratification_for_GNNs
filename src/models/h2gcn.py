import torch
import torch.nn as nn
import torch.nn.functional as F


class H2GCN_2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        H2GCN-2 Model (K=2 rounds of aggregation)
        Based on the config: M64-R-T1-G-V-T2-G-V-C1-C2-D0.5-MO
        """
        super(H2GCN_2, self).__init__()

        # Stage 1: Feature Embedding (M64-R)
        self.embed = nn.Linear(in_channels, hidden_channels, bias=False)
        self.dropout = 0.5

        # Stage 3: Classification (MO)
        # We must calculate the size of the final concatenated representation (C1-C2).
        # - Round 0 (r0) size: hidden_dim (64)
        # - Round 1 (r1) size: hidden_dim * 2 (1-hop + 2-hop concatenation) = 128
        # - Round 2 (r2) size: r1_size * 2 = 256
        # Total concatenated size = 64 + 128 + 256 = 448
        final_dim = hidden_channels + (hidden_channels * 2) + (hidden_channels * 4)

        self.classifier = nn.Linear(final_dim, out_channels, bias=False)

    def forward(self, x, adj_dict):
        """
        x: Input node features (N x in_features)
        adj_hop1: Normalized 1-hop sparse adjacency matrix (torch.sparse.FloatTensor)
        adj_hop2: Normalized 2-hop sparse adjacency matrix (torch.sparse.FloatTensor)
        """
        adj_hop1, adj_hop2 = adj_dict['adj1_hop'], adj_dict['adj2_hop']
        # --- Stage 1: Feature Embedding ---
        # M64-R -> Tagged as T1 (r0)
        r0 = F.relu(self.embed(x))

        # --- Stage 2: Neighborhood Aggregation ---
        # Round 1 (G-V): Aggregate 1-hop and 2-hop independently, then Vectorize (concat)
        r1_hop1 = torch.sparse.mm(adj_hop1, r0)
        r1_hop2 = torch.sparse.mm(adj_hop2, r0)
        r1 = torch.cat([r1_hop1, r1_hop2], dim=1)  # Tagged as T2 (r1)

        # Round 2 (G-V): Aggregate based on the output of Round 1
        r2_hop1 = torch.sparse.mm(adj_hop1, r1)
        r2_hop2 = torch.sparse.mm(adj_hop2, r1)
        r2 = torch.cat([r2_hop1, r2_hop2], dim=1)

        # --- Stage 3: Classification ---
        # C1-C2 in the reference appends T1, then T2 to the current representation.
        r_final = torch.cat([r2, r0, r1], dim=1)

        # D0.5: Dropout
        r_final = F.dropout(r_final, p=self.dropout, training=self.training)

        # MO: Model Output. The project training loop uses F.nll_loss, so return
        # log-probabilities; this is equivalent to the reference softmax CE.
        out = self.classifier(r_final)

        return F.log_softmax(out, dim=1)
