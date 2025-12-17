"""
KNN classification using engineered features from feature_data/full_features.csv
"""


import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# PATHS
DATA_DIR = Path("preprocessed_data/full")
METADATA_FILE = DATA_DIR / "labels_and_metadata.csv"
RAW_DATA_DIR = Path("data")
FEATURES_FILE = Path("feature_data/full_features.csv")

# CONFIGURATIONS
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_NEIGHBORS = 31
K_FOLDS = 5
PLOT_CONFUSION_MATRIX = True
PLOT_K_ANALYSIS = True
PRINT_RESULTS = True

k_values = [1, 3, 5, 7, 9, 11, 15, 17, 21, 25, 27, 30, 31, 32, 35, 41, 45, 51]
k_results = {k: {'accuracies': [], 'f1s': [], 'y_true': [], 'y_pred': []} for k in k_values}


# LOAD DATA

print("Loading feature data...")
df = pd.read_csv(FEATURES_FILE)

# Separate features from metadata
metadata_cols = ['filename', 'label', 'original_file']
feature_cols = [col for col in df.columns if col not in metadata_cols]

print(f"Total features: {len(feature_cols)}")
print(f"Total samples: {len(df)}")
print(f"  Undrugged (label=0): {(df['label']==0).sum()}")
print(f"  Drugged (label=1): {(df['label']==1).sum()}")

# Extract features and labels
X = df[feature_cols].values
y = df['label'].values
worm_ids = df['original_file'].values

# Handle any NaN/inf values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print(f"\nFeature matrix shape: {X.shape}")
print(f"  {X.shape[0]} worms, {X.shape[1]} features")


# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# CROSS-VALIDATION & K-VALUE ANALYSIS

print("\nRunning cross-validation...")

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


# RESULTS

if PRINT_RESULTS:
    print("\n" + "=" * 50)
    print("K-VALUE ANALYSIS")
    print("=" * 50)

    k_summary = []
    for k in k_values:
        # Skip K values that have no results
        if len(k_results[k]['accuracies']) == 0:
            print(f"  K={k:2d} Skipped (too large for training set)")
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
        print(f"  K={k:2d} Acc={mean_acc:.4f}±{std_acc:.4f}, F1={mean_f1:.4f}+-{std_f1:.4f}")


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
    plt.title(f'Feature-based KNN Confusion Matrix (k={N_NEIGHBORS})')
    plt.tight_layout()
    plt.savefig('KNN/features_knn_confusion_matrix.pdf', dpi=300)

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
    plt.savefig('KNN/features_knn_k_analysis.pdf', dpi=300)
