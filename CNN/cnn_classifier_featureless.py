"""
DECLARATION: Parts of this code written with help from Claude Sonnet 4.5 (Anthropic, 2025)

CNN classification
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


# PATHS
DATA_DIR = Path("../preprocessed_data/full")
METADATA_FILE = DATA_DIR / "labels_and_metadata.csv"
RAW_DATA_DIR = Path("../data")

# CONFIGURATIONS
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

K_FOLDS = 5
ADDITIONAL_DATA = True
DOWNSAMPLE_FACTOR = 3  # Take every 3rd point

PLOT_CONFUSION_MATRIX = True
PRINT_RESULTS = True

# CNN hyperparameters
EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 0.001
PATIENCE = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# CNN MODEL

class CNN1D(nn.Module):
    """1D CNN with moderate complexity for speed/performance balance"""
    def __init__(self):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(4)
        self.drop1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.drop2 = nn.Dropout(0.3)
        
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(4)
        self.drop3 = nn.Dropout(0.3)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(128, 64)
        self.drop4 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.drop1(self.pool1(self.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(self.relu(self.bn3(self.conv3(x)))))
        
        x = self.global_pool(x).squeeze(-1)
        
        x = self.drop4(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        
        return x


# LOAD DATA

def load_time_series(file_path):
    """Load and downsample time series."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()
    
    speed = df['speed'].values
    speed = pd.Series(speed).fillna(method='ffill').fillna(0).values
    
    # Downsample to reduce length
    speed = speed[::DOWNSAMPLE_FACTOR]
    
    return speed


print("Loading time series data.")
X_list = []
y_list = []
worm_ids = []

metadata = pd.read_csv(METADATA_FILE)

for idx, row in metadata.iterrows():
    relative_path = row['relative_path']
    file_path = DATA_DIR / relative_path / row['file']
    
    try:
        ts = load_time_series(file_path)
        X_list.append(ts)
        y_list.append(row['label'])
        worm_ids.append(row['original_file'])
        
        if (idx + 1) % 20 == 0:
            print(f"  Loaded {idx + 1}/{len(metadata)} worms")
    except Exception as e:
        print(f"  Warning: Could not load {file_path}: {e}")


if ADDITIONAL_DATA:
    additional_folders = [
        ('NoTerbinafine', 0),
        ('Terbinafine', 1)
    ]
    
    for folder_name, label in additional_folders:
        folder_path = RAW_DATA_DIR / folder_name
        if folder_path.exists():
            csv_files = list(folder_path.glob("*.csv"))
            print(f"  Found {len(csv_files)} files in {folder_name} (label={label})")
            
            for csv_file in csv_files:
                try:
                    ts = load_time_series(csv_file)
                    X_list.append(ts)
                    y_list.append(label)
                    worm_ids.append(csv_file.stem)
                except Exception as e:
                    print(f"    Warning: Could not load {csv_file.name}: {e}")


# Pad sequences
max_len = min(len(x) for x in X_list)
print(f"\nPadding sequences to length: {max_len}")

X_padded = []
for ts in X_list:
    if len(ts) > max_len:
        ts = ts[:max_len]
    elif len(ts) < max_len:
        ts = np.pad(ts, (0, max_len - len(ts)), 'constant')
    X_padded.append(ts)

X = np.array(X_padded)
y = np.array(y_list)
worm_ids = np.array(worm_ids)

X = X.reshape(X.shape[0], 1, X.shape[1])  # (samples, channels, timesteps)

print(f"Final dataset shape: {X.shape}")
print(f"  {X.shape[0]} worms, {X.shape[2]} timesteps")
print(f"  Undrugged: {(y==0).sum()}, Drugged: {(y==1).sum()}")


# Normalize
scaler = StandardScaler()
for i in range(len(X)):
    X[i, 0, :] = scaler.fit_transform(X[i, 0, :].reshape(-1, 1)).flatten()


# TRAINING FUNCTIONS

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += y_batch.size(0)
        correct += (predicted.squeeze() == y_batch).sum().item()
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predicted = (outputs > 0.5).float()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    return np.array(all_preds).flatten(), np.array(all_labels)


# CROSS-VALIDATION

print("\n" + "CNN cross validation results:")

gkf = GroupKFold(n_splits=K_FOLDS)
fold_results = []

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=worm_ids)):
    print(f"\n--- Fold {fold_idx + 1}/{K_FOLDS} ---")
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"  Train: {len(X_train)} worms | Test: {len(X_test)} worms")
    print(f"  Train balance: [0={(y_train==0).sum()}, 1={(y_train==1).sum()}]")
    print(f"  Test balance: [0={(y_test==0).sum()}, 1={(y_test==1).sum()}]")
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    model = CNN1D().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        
        # Evaluate every 3 epochs
        if (epoch + 1) % 3 == 0:
            y_pred, _ = evaluate(model, test_loader)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            print(f"    Epoch {epoch+1}/{EPOCHS}: Loss={train_loss:.4f}, TrainAcc={train_acc:.4f}, TestF1={f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    
    # Final evaluation
    y_pred, _ = evaluate(model, test_loader)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    
    pred_counts = np.bincount(y_pred.astype(int), minlength=2)
    print(f"  Predictions: [0={pred_counts[0]}, 1={pred_counts[1]}]")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    fold_results.append({
        'fold': fold_idx + 1,
        'accuracy': acc,
        'f1': f1,
        'y_true': y_test,
        'y_pred': y_pred
    })


# RESULTS

print("\n" + "CNN results:")

all_y_true = np.concatenate([r['y_true'] for r in fold_results])
all_y_pred = np.concatenate([r['y_pred'] for r in fold_results])

mean_acc = np.mean([r['accuracy'] for r in fold_results])
std_acc = np.std([r['accuracy'] for r in fold_results])
mean_f1 = np.mean([r['f1'] for r in fold_results])
std_f1 = np.std([r['f1'] for r in fold_results])

print(f"\nPerformance:")
print(f"  Accuracy: {mean_acc:.4f} +- {std_acc:.4f}")
print(f"  F1 Score: {mean_f1:.4f} +- {std_f1:.4f}")

print("\nClassification Report:")
print(classification_report(all_y_true, all_y_pred.astype(int), target_names=['Undrugged', 'Drugged']))


# CONFUSION MATRIX

if PLOT_CONFUSION_MATRIX:
    cm = confusion_matrix(all_y_true, all_y_pred.astype(int))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Undrugged', 'Drugged'],
                yticklabels=['Undrugged', 'Drugged'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('CNN Confusion Matrix')
    plt.tight_layout()
    plt.savefig('cnn_confusion_matrix.png', dpi=300)
    print("\nConfusion matrix saved to CNN/cnn_confusion_matrix.png")