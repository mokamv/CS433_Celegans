import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("SIMPLE FEATURELESS KNN CLASSIFICATION")
print("(Using flattened time series with Euclidean distance)")
print("=" * 70)

# Configuration

DATA_DIR = Path("data_google_drive/preprocessed_data/full")
METADATA_FILE = DATA_DIR / "labels_and_metadata.csv"
DOWNSAMPLE_FACTOR = 1 # Take every 50th point for speed
N_NEIGHBORS = 15

print(f"\nConfiguration:")
print(f"  Downsample factor: {DOWNSAMPLE_FACTOR} (smaller = more data)")
print(f"  K neighbors: {N_NEIGHBORS}")

# Load metadata
metadata = pd.read_csv(METADATA_FILE)
print(f"\nLoaded {len(metadata)} worms")
print(f"  Undrugged (label=0): {(metadata['label']==0).sum()}")
print(f"  Drugged (label=1): {(metadata['label']==1).sum()}")


# Load and flatten time series data

def load_and_flatten(file_path, downsample):
    """Load time series and flatten to a feature vector."""
    df = pd.read_csv(file_path)
    
    # Use speed, x, y
    features = ['speed', 'x', 'y']
    ts_data = df[features].values
    
    # Handle NaN
    ts_data = pd.DataFrame(ts_data).fillna(method='ffill').fillna(0).values
    
    # Downsample
    ts_data = ts_data[::downsample]
    
    # Flatten to 1D vector
    return ts_data.flatten()

print("\nLoading and flattening time series...")
X_list = []
y_list = []
worm_ids = []

for idx, row in metadata.iterrows():
    # Fix path
    relative_path = row['relative_path']
    if 'TERBINAFINE- (control)-' in relative_path:
        relative_path = 'TERBINAFINE- (control)'
    
    file_path = DATA_DIR / relative_path / row['file']
    
    try:
        vec = load_and_flatten(file_path, DOWNSAMPLE_FACTOR)
        X_list.append(vec)
        y_list.append(row['label'])
        worm_ids.append(row['original_file'])
        
        if (idx + 1) % 20 == 0:
            print(f"  Loaded {idx + 1}/{len(metadata)} worms...")
    except Exception as e:
        print(f"  Warning: Could not load {file_path}: {e}")

# Pad all vectors to same length
max_len = max(len(x) for x in X_list)
print(f"\nPadding vectors to length: {max_len}")

X_padded = []
for vec in X_list:
    if len(vec) < max_len:
        vec = np.pad(vec, (0, max_len - len(vec)), 'constant')
    else:
        vec = vec[:max_len]
    X_padded.append(vec)

X = np.array(X_padded)
y = np.array(y_list)
worm_ids = np.array(worm_ids)

print(f"Final dataset shape: {X.shape}")
print(f"  {X.shape[0]} worms, {X.shape[1]} features per worm")


# Normalize features

print("\nNormalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Cross-validation

print("\n" + "=" * 70)
print("CROSS-VALIDATION (Worm-level)")
print("=" * 70)

gkf = GroupKFold(n_splits=5)
fold_results = []

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y, groups=worm_ids)):
    print(f"\n--- Fold {fold_idx + 1}/5 ---")
    
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"  Train: {len(X_train)} worms | Test: {len(X_test)} worms")
    print(f"  Train balance: {np.bincount(y_train)}")
    print(f"  Test balance: {np.bincount(y_test)}")
    
    # Train KNN
    clf = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    fold_results.append({
        'fold': fold_idx + 1,
        'accuracy': acc,
        'f1': f1,
        'y_true': y_test,
        'y_pred': y_pred
    })
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")


# Aggregate results

print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

all_y_true = np.concatenate([r['y_true'] for r in fold_results])
all_y_pred = np.concatenate([r['y_pred'] for r in fold_results])

mean_acc = np.mean([r['accuracy'] for r in fold_results])
std_acc = np.std([r['accuracy'] for r in fold_results])
mean_f1 = np.mean([r['f1'] for r in fold_results])
std_f1 = np.std([r['f1'] for r in fold_results])

print(f"\nWorm-level Performance:")
print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
print(f"  F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")

print("\nClassification Report:")
if len(np.unique(all_y_true)) > 1:
    print(classification_report(all_y_true, all_y_pred, 
                              target_names=['Undrugged', 'Drugged']))


# Confusion Matrix
cm = confusion_matrix(all_y_true, all_y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Undrugged', 'Drugged'],
            yticklabels=['Undrugged', 'Drugged'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Featureless KNN Confusion Matrix (k={N_NEIGHBORS})')
plt.tight_layout()
plt.savefig('Figures/featureless_knn_confusion_matrix.png', dpi=300)



# Test different K values
print("\n" + "=" * 70)
print("K-VALUE ANALYSIS")
print("=" * 70)

k_values = [1, 3, 5, 7, 9, 11, 15]
k_results = []

train_idx, test_idx = next(gkf.split(X_scaled, y, groups=worm_ids))
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

for k in k_values:
    print(f"  K={k}...", end=' ')
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    k_results.append({'k': k, 'accuracy': acc, 'f1': f1})
    print(f"Acc={acc:.4f}, F1={f1:.4f}")

# Plot K analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

k_vals = [r['k'] for r in k_results]
accs = [r['accuracy'] for r in k_results]
f1s = [r['f1'] for r in k_results]

ax1.plot(k_vals, accs, 'o-', linewidth=2, markersize=8)
ax1.set_xlabel('K (Number of Neighbors)')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy vs K')
ax1.grid(True, alpha=0.3)

ax2.plot(k_vals, f1s, 'o-', linewidth=2, color='orange')
ax2.set_xlabel('K (Number of Neighbors)')
ax2.set_ylabel('F1 Score')
ax2.set_title('F1 Score vs K')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Figures/featureless_knn_k_analysis.png', dpi=300)

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nFeatureless classification using simple KNN")
print(f"F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
print("=" * 70)
