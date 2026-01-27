import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

def create_sample_specific_graphs(X, num_nodes=128, top_k_edges=256):
    """
    Creates sample-specific graphs using attention-inspired top-K edge selection.
    Each sample is projected into node space and fully attended to itself.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    linear_layer = nn.Linear(X.shape[1], num_nodes).to(device)
    query_layer = nn.Linear(1, 32).to(device)
    key_layer = nn.Linear(1, 32).to(device)

    graph_data_list = []
    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        for i in range(X_tensor.shape[0]):
            x = linear_layer(X_tensor[i:i+1]).squeeze(0)
            x = x.unsqueeze(-1)

            q = query_layer(x).squeeze(-2)
            k = key_layer(x).squeeze(-2)

            attn = torch.matmul(q, k.T) / (32 ** 0.5)
            attn = F.softmax(attn, dim=-1)

            topk_vals, topk_idx = torch.topk(attn.view(-1), top_k_edges)
            row = topk_idx // num_nodes
            col = topk_idx % num_nodes

            edge_index = torch.stack([row, col], dim=0)

            graph_data_list.append(Data(x=x, edge_index=edge_index))

    return graph_data_list
