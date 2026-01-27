import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GNNOnlyModel(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=64,
        output_dim=512,
        num_classes=4,
        num_heads=4,
        dropout=0.4
    ):
        super().__init__()

        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)

        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim * num_heads)

        self.fc = nn.Linear(hidden_dim * num_heads, output_dim)
        self.out = nn.Linear(output_dim, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.gat1(x, edge_index)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.gat2(x, edge_index)))
        x = self.dropout(x)

        x = global_mean_pool(x, batch)

        x = F.relu(self.fc(x))
        x = self.dropout(x)

        return self.out(x)
