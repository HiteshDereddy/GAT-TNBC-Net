import torch
from torch_geometric.data import Batch

def custom_collate(batch):
    graphs = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return Batch.from_data_list(graphs), labels
