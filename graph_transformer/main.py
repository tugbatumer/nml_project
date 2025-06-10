from pathlib import Path
import numpy as np
import pandas as pd
from seiz_eeg.dataset import EEGDataset
from torch.utils.data import random_split
import torch
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader
import argparse


import utils
from transformer_model import GraphTransformer
from helpers import seed_everything, build_softmax_thresholded_graph, handcrafted_features_combined, handcrafted_features, fft_filtering_features, batch_to_dense_E
from placeholder import PlaceHolder
from eeggraphdataset import EEGGraphDataset

# Example usage: python main.py --n_layers 3 --n_heads 8 --lr 1e-4  --batch_size 256 --threshold 0.25 --epochs 500 --save --submit
parser = argparse.ArgumentParser(description="Train GraphTransformer on EEG data")
parser.add_argument("--n_layers",    type=int,   default=2,       help="number of Transformer layers")
parser.add_argument("--n_heads",     type=int,   default=4,       help="number of attention heads")
parser.add_argument("--lr",          type=float, default=1e-4,    help="learning rate")
parser.add_argument("--batch_size",  type=int,   default=512,     help="batch size")
parser.add_argument("--epochs",      type=int,   default=1000,    help="number of training epochs")
parser.add_argument("--threshold",   type=float, default=0.25,    help="prediction threshold for positive class")
parser.add_argument("--submit",      action="store_true",        help="if set, prepare a submission at the end")
parser.add_argument("--save",      action="store_true",        help="if set, save the trained model")

args = parser.parse_args()

# Then replace your vars:
n_layers   = args.n_layers
n_heads    = args.n_heads
lr         = args.lr
batch_size = args.batch_size
epochs     = args.epochs
threshold  = args.threshold
submit     = args.submit
save       = args.save

# we can also try hidden dim params

seed_everything(21)

data_path = "/home/ogut/data/"

DATA_ROOT = Path(data_path)
DATASET_ROOT = Path(f"{data_path}/")


#### BUILD GRAPH ###

# Load your CSV file
df = pd.read_csv("distances_3d.csv")  # Update with your actual file path


# Get sorted list of unique node names (e.g., 'FP1', 'F3', ...)
nodes = sorted(set(df["from"]) | set(df["to"]))
node_to_idx = {name: idx for idx, name in enumerate(nodes)}

# Create empty matrix
N = len(nodes)
dist = np.full((N, N), np.inf)  # Initialize with inf (or large number)

# Fill matrix
for _, row in df.iterrows():
    i = node_to_idx[row["from"]]
    j = node_to_idx[row["to"]]
    dist[i, j] = float(row["distance"])

edge_index, edge_attr = build_softmax_thresholded_graph(dist, beta=5, keep_ratio=0.9)

preprocess_method = fft_filtering_features

### LOAD TRAIN DATA ###
clips_tr = pd.read_parquet(DATASET_ROOT / "train/segments.parquet")
dataset_tr = EEGDataset(
    clips_tr,
    signals_root=DATASET_ROOT / "train",
    signal_transform=preprocess_method,
    prefetch=True,  # If your compute does not allow it, you can use `prefetch=False`
)

# Split with no patients being in both train and validation split
def extract_patient(idx):
    # if the index entry is a tuple (e.g. a MultiIndex), grab its first element
    first = idx[0] if isinstance(idx, tuple) else idx
    # now split on '_' and take the patient prefix
    return first.split('_')[0]

# apply that to the index
clips_tr = clips_tr.copy()
clips_tr['patient'] = clips_tr.index.map(extract_patient)

# now split patients
unique_pats = clips_tr['patient'].unique()
shuffled = np.random.permutation(unique_pats)
val_pats   = shuffled[:21]
train_pats = shuffled[21:]

train_df = clips_tr[clips_tr['patient'].isin(train_pats)].copy()
val_df   = clips_tr[clips_tr['patient'].isin(val_pats)].copy()

print(f"Train patients: {train_df['patient'].nunique()}, samples: {len(train_df)}")
print(f"  Val patients: {val_df  ['patient'].nunique()}, samples: {len(val_df)  }")

train_split = EEGDataset(
    train_df,
    signals_root=DATA_ROOT / "train",
    signal_transform=preprocess_method,
    prefetch=True,  # If your compute does not allow it, you can use `prefetch=False`
)
val_split = EEGDataset(
    val_df,
    signals_root=DATA_ROOT / "train",
    signal_transform=preprocess_method,
    prefetch=True,  # If your compute does not allow it, you can use `prefetch=False`
)

# Now wrap both in EEGGraphDataset
train_dataset = EEGGraphDataset(train_split, edge_index=edge_index, edge_attr=edge_attr, is_train=True)
val_dataset = EEGGraphDataset(val_split, edge_index=edge_index, edge_attr=edge_attr, is_train=True)

### PREPARE MODEL ###

# 1) Number of “graph‐nodes” = N = 19
N = train_dataset.in_dim                # 19

# 2) True “features per node” = F, taken from raw EEGDataset:
#    dataset_tr[i][0] has shape (19, F), so F = dataset_tr[0][0].shape[1].
F = dataset_tr[0][0].shape[1]           # e.g. 372

in_feat_per_node  = F                   # 372
in_feat_per_edge  = 1
in_feat_global    = 1                   # no real globals→feed zeros

# 3) Hidden dims (you can tune)
hidden_dims = {
    "X":  64,
    "E":  64,
    "y":  64,
    "dx": 64,
    "de": 64,
    "dy": 64,
}

input_dims  = {"X": in_feat_per_node, "E": in_feat_per_edge, "y": in_feat_global}
output_dims = {"X": 0,              "E": 0,             "y": 1}  # only 1 logit


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = GraphTransformer(
    n_layers=n_layers,
    n_head=n_heads,
    input_dims=input_dims,
    hidden_dims=hidden_dims,
    output_dims=output_dims,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr) #was 1e-3
criterion = torch.nn.BCEWithLogitsLoss()


### TRAIN & VAL ###

# DataLoaders (use PyG’s DataLoader)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)


# Grab the single‐graph adjacency for all batches
global_edge_index = train_dataset.edge_index.to(device)   # shape (2, M)
global_edge_attr  = train_dataset.edge_attr.to(device)    # shape (M,)

train_losses, train_accs, train_f1scores = [], [], []
val_losses,   val_accs,   val_f1scores   = [], [], []

fname = f"outputs/results_{preprocess_method.__name__}__layers{n_layers}_heads{n_heads}_lr{lr:.0e}_bs{batch_size}.txt"
open(fname, "w").close()

for epoch in range(epochs):
    model.train()
    all_train_preds, all_train_labels = [], []
    total_train_loss = 0.0

    for batch in train_loader:
        # `batch` is a PyG Batch of Data(x, edge_index, edge_attr, y)
        batch = batch.to(device)

        bs = batch.y.size(0)   # number of graphs in this batch

        # ── a) Reshape node‐features from (bs*F, 19) → (bs, 19, F)
        # Note: batch.x originally is stacked (F,19) per graph → (bs*F, 19)
        temp    = batch.x.view(bs, F, N)    # shape (bs, F, 19)
        X_batch = temp.permute(0, 2, 1)     # shape (bs, 19, F)

        # ── b) Dense edge‐tensor (same adjacency for each graph)
        E_batch = batch_to_dense_E(global_edge_index, global_edge_attr, bs, N)
        # → shape (bs, 19, 19, 1)

        # ── c) Global “y” = zeros
        y_global_in = torch.zeros((bs, in_feat_global), device=device)

        # ── d) Node mask = all‐ones
        node_mask = torch.ones((bs, N), dtype=torch.bool, device=device)

        # ── e) Pack into PlaceHolder & forward
        holder_in  = PlaceHolder(X=X_batch, E=E_batch, y=y_global_in)
        out_holder = model(holder_in, node_mask)
        # out_holder.y has shape (bs, 1)

        logits = out_holder.y.view(bs)           # shape (bs,)
        labels = batch.y.view(bs).float()        # shape (bs,)

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        # record train preds & labels
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        all_train_preds .extend(preds.cpu().tolist())
        all_train_labels.extend(labels.cpu().tolist())

    # epoch‐level train metrics
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    train_acc = (np.array(all_train_preds) == np.array(all_train_labels)).mean()
    train_accs.append(train_acc)

    train_f1 = f1_score(all_train_labels, all_train_preds)
    train_f1scores.append(train_f1)
    

    
    # Validation (compute F1)
    
    model.eval()
    all_preds  = []
    all_labels = []

    total_val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            bs = batch.y.size(0)

            # a) Reshape node‐features again
            temp    = batch.x.view(bs, F, N)
            X_batch = temp.permute(0, 2, 1)   # (bs, 19, F)

            # b) Dense edge tensor
            E_batch = batch_to_dense_E(global_edge_index, global_edge_attr, bs, N)

            # c) Global zeros + node_mask
            y_global_in = torch.zeros((bs, in_feat_global), device=device)
            node_mask   = torch.ones((bs, N), dtype=torch.bool, device=device)

            # d) Forward
            holder_in  = PlaceHolder(X=X_batch, E=E_batch, y=y_global_in)
            out_holder = model(holder_in, node_mask)

            logits = out_holder.y.view(bs)
            labels = batch.y.view(bs).float()
            loss = criterion(logits, labels)
            total_val_loss += loss.item()

            probs  = torch.sigmoid(logits)
            preds  = (probs > threshold).float()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch.y.view(bs).cpu().tolist())

    # epoch‐level val metrics
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    val_accs.append(val_acc)

    val_f1 = f1_score(all_labels, all_preds)
    val_f1scores.append(val_f1)

    msg = (
        f"Epoch {epoch+1:03d} | "
        f"Tr L={avg_train_loss:.4f} Acc={train_acc:.4f} F1={train_f1:.4f} | "
        f"Val L={avg_val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f}"
    )
    print(msg)

    # append to file
    with open(fname, "a") as f:
        f.write(msg + "\n")

### OPTIONAL SAVE ###
if save:
    save_path = f"models/graph_transformer_{preprocess_method.__name__}__epochs{epochs}_layers{n_layers}_heads{n_heads}_lr{lr:.0e}_bs{batch_size}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

### OPTIONAL SUBMISSION
if submit:

    # Create test dataset
    clips_te = pd.read_parquet(DATASET_ROOT / "test/segments.parquet")
    dataset_te = EEGDataset(
        clips_te,  # Your test clips variable
        signals_root=DATA_ROOT
        / "test",  # Update this path if your test signals are stored elsewhere
        signal_transform=preprocess_method,  # You can change or remove the signal_transform as needed
        prefetch=True,  # Set to False if prefetching causes memory issues on your compute environment
        return_id=True,  # Return the id of each sample instead of the label
    )
    test_dataset = EEGGraphDataset(dataset_te, edge_index=edge_index, edge_attr=edge_attr, is_train=False)

    # Create DataLoader for the test dataset
    loader_te  = DataLoader(test_dataset,   batch_size=32, shuffle=False)

    model.eval()
    all_ids   = []
    all_preds = []

    with torch.no_grad():
        for batch in loader_te:
            # batch is a Batch(Data, …), with attributes:
            #   batch.x         # Tensor of shape (bs*19, feature_dim)
            #   batch.edge_index, batch.edge_attr
            #   batch.id        # list of sample‐ID strings
            #   batch.batch     # graph‐index for each node

            # 1) grab IDs
            ids = batch.id
            # (if for some reason batch.id is a tensor, do: ids = batch.id.tolist())

            batch = batch.to(device)
            
            # 2) move the *tensors* to device
            x_flat      = batch.x.to(device)

            bs = batch.num_graphs  # number of EEG clips in this batch

            # 3) un‐flatten your node features exactly like in train:
            temp    = x_flat.view(bs, F, N)      # (bs, F, 19)
            X_batch = temp.permute(0, 2, 1)      # (bs, 19, F)

            # 4) build dense edges
            E_batch = batch_to_dense_E(global_edge_index, global_edge_attr, bs, N)

            # 5) dummy global + node_mask
            y_global_in = torch.zeros((bs, in_feat_global), device=device)
            node_mask   = torch.ones((bs, N), dtype=torch.bool, device=device)

            # 6) forward
            holder_in  = PlaceHolder(X=X_batch, E=E_batch, y=y_global_in)
            out_holder = model(holder_in, node_mask)
            logits     = out_holder.y.view(bs)

            # 7) threshold
            preds = (logits > 0).long().cpu().tolist()

            # 8) collect
            all_preds.extend(preds)
            all_ids.extend(ids)

            submission_df = pd.DataFrame({"id": all_ids, "label": all_preds})
            submission_df.to_csv(f"predictions/submission_{preprocess_method.__name__}__epochs{epochs}_layers{n_layers}_heads{n_heads}_lr{lr:.0e}_bs{batch_size}.csv", index=False)




