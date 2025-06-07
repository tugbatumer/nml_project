import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class EEGGraphDataset(Dataset):
    def __init__(self, data, edge_index, edge_attr, is_train=True):
        super().__init__()
        self.num_samples = len(data)
        self.in_dim = data[0][0].shape[0]
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.data = data
        self.is_train = is_train

    def __len__(self):
        return self.num_samples

#     def __getitem__(self, idx):
#         x = torch.tensor(self.data[idx][0].T, dtype=torch.float32)  # Simulated signal
#         y = int(self.data[idx][1]) # Target label if training, else ID
#         if self.is_train:
#             return Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr, y=torch.tensor([[y]], dtype=torch.float32))
#         else:
#             return Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr, id=y)
        
    def __getitem__(self, idx):
        # Unpack the raw EEGDataset item:
        #   self.data[idx] is (signal_array, label_or_id)
        x_np, meta = self.data[idx]

        # Build the node‐feature tensor (still (F, N) because you .T below)
        x = torch.tensor(x_np.T, dtype=torch.float32)

        if self.is_train:
            # meta should be numeric (0 or 1)
            label = float(meta)
            return Data(
                x          = x,
                edge_index = self.edge_index,
                edge_attr  = self.edge_attr,
                y          = torch.tensor([label], dtype=torch.float32),
            )
        else:
            # meta is your sample‐ID string
            return Data(
                x          = x,
                edge_index = self.edge_index,
                edge_attr  = self.edge_attr,
                id         = meta,    # no int() here!
            )