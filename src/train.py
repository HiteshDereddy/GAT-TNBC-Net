import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from torch.utils.data import DataLoader
from graph_utils import create_sample_specific_graphs
from model import GNNOnlyModel
from utils import custom_collate

# ------------------ Config ------------------
CSV_PATH = "../data/merged_tpm_zscore_360.csv"
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-5
PATIENCE = 20
# --------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_csv(CSV_PATH).dropna(axis=1, how="all")

X = df.drop(columns=["sample_id", "subtype_old", "subtype_new", "tumor_purity", "Gene1"])
y = df["subtype_new"]

X = X.dropna()
y = y.loc[X.index]

le = LabelEncoder()
y = le.fit_transform(y)

X = StandardScaler().fit_transform(X)

# Graph creation
graphs = create_sample_specific_graphs(X)

# Train/val/test split
X_train, X_tmp, y_train, y_tmp = train_test_split(
    graphs, y, test_size=0.3, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
)

y_train = torch.LongTensor(y_train).to(device)
y_val = torch.LongTensor(y_val).to(device)
y_test = torch.LongTensor(y_test).to(device)

def loader(graphs, labels):
    return DataLoader(
        list(zip(graphs, labels)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate
    )

train_loader = loader(X_train, y_train)
val_loader = loader(X_val, y_val)

model = GNNOnlyModel(num_classes=len(le.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

best_val = float("inf")
patience_ctr = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for g, yb in train_loader:
        g = g.to(device)
        optimizer.zero_grad()
        loss = criterion(model(g.x, g.edge_index, g.batch), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for g, yb in val_loader:
            g = g.to(device)
            val_loss += criterion(model(g.x, g.edge_index, g.batch), yb).item()

    if val_loss < best_val:
        best_val = val_loss
        patience_ctr = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            break

# Evaluation
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

from torch_geometric.data import Batch
test_batch = Batch.from_data_list(X_test).to(device)

with torch.no_grad():
    preds = model(test_batch.x, test_batch.edge_index, test_batch.batch).argmax(dim=1)

print(classification_report(y_test.cpu(), preds.cpu(), target_names=le.classes_))
print(confusion_matrix(y_test.cpu(), preds.cpu()))
