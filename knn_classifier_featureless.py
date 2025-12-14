"""
Featureless KNN classification using flattened time series with Euclidean distance.
ONLY RAWDATA (SPEED IS USED)
"""


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


# PATHS
DATA_DIR = Path("data_google_drive/preprocessed_data/full")
METADATA_FILE = DATA_DIR / "labels_and_metadata.csv"
RAW_DATA_DIR = Path("data_google_drive/data_google_drive")

# CONFIGURATIONS
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_NEIGHBORS = 31 # For plotting confusion matrix
K_FOLDS = 5
PLOT_CONFUSION_MATRIX = True
PLOT_K_ANALYSIS = True
ADDITIONAL_DATA = True  # Load extra data from Raw folder
PRINT_RESULTS = True

features = ['speed'] # NOTE: Additional data does not have turning angle

k_values = [1, 3, 5, 7, 9, 11, 15, 17, 21, 25, 27, 30, 31, 32, 35]
k_results = {k: {'accuracies': [], 'f1s': [], 'y_true': [], 'y_pred': []} for k in k_values}


# LOAD DATA

def load_and_flatten(file_path, features=features):
    """Load time series and flatten to a feature vector."""
    
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower() # Handle both lowercase and uppercase column names
    ts_data = df[features].values
    ts_data = pd.DataFrame(ts_data).fillna(method='ffill').fillna(0).values # Handle NaN
    
    return ts_data.flatten()


X_list = []
y_list = []
worm_ids = []

# Load from metadata
metadata = pd.read_csv(METADATA_FILE)

for idx, row in metadata.iterrows():
    # Fix path
    relative_path = row['relative_path']
    if 'TERBINAFINE- (control)-' in relative_path:
        relative_path = 'TERBINAFINE- (control)'
    
    file_path = DATA_DIR / relative_path / row['file']
    
    try:
        vec = load_and_flatten(file_path)
        X_list.append(vec)
        y_list.append(row['label'])
        worm_ids.append(row['original_file'])
        
        if (idx + 1) % 20 == 0:
            print(f"  Loaded {idx + 1}/{len(metadata)} worms")
    except Exception as e:
        print(f"  Warning: Could not load {file_path}: {e}")


# Load additional data from NoTerbinafine (label=0) and Terbinafine (label=1)

if ADDITIONAL_DATA:
    additional_folders = [
        ('NoTerbinafine', 0),  # Undrugged
        ('Terbinafine', 1)      # Drugged
    ]

    initial_count = len(X_list)

    for folder_name, label in additional_folders:
        folder_path = RAW_DATA_DIR / folder_name
        if folder_path.exists():
            csv_files = list(folder_path.glob("*.csv"))
            print(f"  Found {len(csv_files)} files in {folder_name} (label={label})")
            
            for csv_file in csv_files:
                try:
                    vec = load_and_flatten(csv_file)
                    X_list.append(vec)
                    y_list.append(label)
                    worm_ids.append(csv_file.stem)  # Use filename as worm ID
                except Exception as e:
                    print(f"    Warning: Could not load {csv_file.name}: {e}")
        else:
            print(f"  Warning: {folder_name} not found")

    additional_count = len(X_list) - initial_count


# Pad all vectors to same length
max_len = max(len(x) for x in X_list)
print(f"\nPadding vectors to length: {max_len}")

X_padded = []
for vec in X_list:
    if len(vec) < max_len:
        # if shorter, pad with zeros
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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# CROSS-VALIDATION & K-VALUE ANALYSIS

gkf = GroupKFold(n_splits=K_FOLDS)

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y, groups=worm_ids)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Limit K values to be less than training set size
    max_k_for_fold = len(X_train)
    
    # Test all K values
    for k in k_values:
        # Skip K values that are too large for this fold
        if k >= max_k_for_fold:
            continue
            
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        k_results[k]['accuracies'].append(acc)
        k_results[k]['f1s'].append(f1)
        k_results[k]['y_true'].append(y_test)
        k_results[k]['y_pred'].append(y_pred)


if PRINT_RESULTS:
    print("\n" + "=" * 50)
    print("K-VALUE ANALYSIS")
    print("=" * 50)

    # Aggregate results
    k_summary = []
    for k in k_values:
        # Skip K values that have no results (too large for all folds)
        if len(k_results[k]['accuracies']) == 0:
            print(f"  K={k:2d} Fold too large for training set")
            continue
            
        mean_acc = np.mean(k_results[k]['accuracies'])
        std_acc = np.std(k_results[k]['accuracies'])
        mean_f1 = np.mean(k_results[k]['f1s'])
        std_f1 = np.std(k_results[k]['f1s'])
        
        k_summary.append({
            'k': k, 
            'accuracy': mean_acc, 
            'acc_std': std_acc,
            'f1': mean_f1,
            'f1_std': std_f1
        })
        print(f"  K={k:2d} Acc={mean_acc:.4f}±{std_acc:.4f}, F1={mean_f1:.4f}±{std_f1:.4f}")
    
    
# PLOTS

if PLOT_CONFUSION_MATRIX:
    # Get predictions for N_NEIGHBORS
    all_y_true = np.concatenate(k_results[N_NEIGHBORS]['y_true'])
    all_y_pred = np.concatenate(k_results[N_NEIGHBORS]['y_pred'])
    
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

if PLOT_K_ANALYSIS:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    k_vals = [r['k'] for r in k_summary]
    accs = [r['accuracy'] for r in k_summary]
    f1s = [r['f1'] for r in k_summary]
    acc_stds = [r['acc_std'] for r in k_summary]
    f1_stds = [r['f1_std'] for r in k_summary]

    ax1.errorbar(k_vals, accs, yerr=acc_stds, fmt='o-', linewidth=2, markersize=8, capsize=5)
    ax1.set_xlabel('K (Number of Neighbors)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs K (mean ± std across folds)')
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(k_vals, f1s, yerr=f1_stds, fmt='o-', linewidth=2, markersize=8, color='orange', capsize=5)
    ax2.set_xlabel('K (Number of Neighbors)')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score vs K (mean ± std across folds)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Figures/featureless_knn_k_analysis.png', dpi=300)