import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
CSV_PATH = "feature_data/segments_features.csv"   # or full_features.csv

WORM_COL  = "original_file"   # <-- USE FILENAME AS WORM ID
LABEL_COL = "label"

N_SPLITS = 5
CORR_THRESHOLD = 0.95   # highly correlated feature filter
VAR_THRESHOLD = 1e-4   # near-constant filter

RANDOM_STATE = 42

# Segment selection
LAST_N_SEGMENTS = None   # set to None to use all segments

# =========================
# Drop non-feature columns
# =========================

def detect_and_drop_non_features(df, label_col, group_col):
    """
    Identify and drop non-feature columns:
    - label
    - group (filename)
    - non-numeric columns
    - index-like numeric columns
    """

    drop_cols = set([label_col, group_col])

    # --- Drop non-numeric columns ---
    for col in df.columns:
        if col in drop_cols:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            drop_cols.add(col)

    # --- Drop index-like numeric columns ---
    for col in df.columns:
        if col in drop_cols:
            continue

        values = df[col].values

        # strictly increasing integers (e.g. 0,1,2,... or 1,2,3,...)
        if np.all(np.diff(values) == 1) or np.all(np.diff(values) == 0):
            drop_cols.add(col)
            continue

        # looks like an index: small integer range == number of rows
        if (
            np.all(values.astype(int) == values)
            and len(np.unique(values)) == len(values)
            and values.min() in [0, 1]
            and values.max() == len(values) - 1
        ):
            drop_cols.add(col)

    return df.drop(columns=list(drop_cols)), sorted(drop_cols)


# =========================
# Utility functions
# =========================
def permutation_importance_worm_level(
    model,
    X_test,
    y_test,
    worms_test,
    baseline_auc,
    n_repeats=5,
    random_state=42
):
    """
    Compute permutation importance at worm level using ROC-AUC.
    """
    rng = np.random.default_rng(random_state)
    n_features = X_test.shape[1]

    importances = np.zeros(n_features)

    for j in range(n_features):
        auc_drops = []

        for _ in range(n_repeats):
            X_perm = X_test.copy()
            rng.shuffle(X_perm[:, j])

            y_pred_perm = model.predict_proba(X_perm)[:, 1]
            y_true_w, y_pred_w = aggregate_by_worm(
                y_test, y_pred_perm, worms_test
            )

            auc_perm = roc_auc_score(y_true_w, y_pred_w)
            auc_drops.append(baseline_auc - auc_perm)

        importances[j] = np.mean(auc_drops)

    return importances

def remove_correlated_features(X, threshold):
    """
    Remove features with pairwise correlation above threshold.
    Keeps the first occurrence.
    """
    corr = np.corrcoef(X, rowvar=False)
    upper = np.triu(np.abs(corr), k=1)

    to_drop = set()
    for i in range(upper.shape[0]):
        for j in range(i + 1, upper.shape[1]):
            if upper[i, j] > threshold:
                to_drop.add(j)

    keep_idx = [i for i in range(X.shape[1]) if i not in to_drop]
    return X[:, keep_idx], keep_idx


def aggregate_by_worm(y_true, y_pred, worm_ids):
    """
    Aggregate segment-level predictions to worm-level
    using mean probability.
    """
    df = pd.DataFrame({
        "worm": worm_ids,
        "y_true": y_true,
        "y_pred": y_pred
    })

    grouped = df.groupby("worm").mean()
    y_true_worm = grouped["y_true"].values
    y_pred_worm = grouped["y_pred"].values

    return y_true_worm, y_pred_worm

def keep_last_n_segments_per_worm(df, worm_col, n_segments):
    """
    Keep only the last n_segments per worm.
    Assumes df is already sorted in time per worm.
    """
    if n_segments is None:
        return df

    return (
        df
        .groupby(worm_col, group_keys=False)
        .tail(n_segments)
        .reset_index(drop=True)
    )


# =========================
# Load data
# =========================
df = pd.read_csv(CSV_PATH)

# OPTIONAL: sanity check ordering
# df = df.sort_values([WORM_COL, "segment_index"])  # only if you have such a column

# Keep only last N segments per worm
df = keep_last_n_segments_per_worm(
    df,
    worm_col=WORM_COL,
    n_segments=LAST_N_SEGMENTS
)

print(f"After segment selection: {len(df)} rows")

# Group labels (filename = worm ID)
worms = df[WORM_COL].values
y = df[LABEL_COL].values

# Drop non-feature columns automatically
df_features, dropped_cols = detect_and_drop_non_features(
    df,
    label_col=LABEL_COL,
    group_col=WORM_COL
)

X = df_features.values
feature_names = df_features.columns
print(f"Dropped non-feature columns: {dropped_cols}")


# =========================
# Cross-validation
# =========================
gkf = GroupKFold(n_splits=N_SPLITS)

all_y_true = []
all_y_pred = []
feature_importances_folds = []


for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups=worms), 1):

    print(f"\nFold {fold}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    worms_test = worms[test_idx]

    # ---- 1. Remove near-constant features (fit on train only!)
    vt = VarianceThreshold(VAR_THRESHOLD)
    X_train = vt.fit_transform(X_train)
    X_test = vt.transform(X_test)

    # ---- 2. Remove highly correlated features (train only!)
    X_train, corr_keep_idx = remove_correlated_features(
        X_train, CORR_THRESHOLD
    )
    X_test = X_test[:, corr_keep_idx]

    print(f"Features after filtering: {X_train.shape[1]}")

    # ---- 3. Scale + SVM
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    pipeline.fit(X_train, y_train)

    # ---- 4. Predict segment-level probabilities
    y_pred_seg = pipeline.predict_proba(X_test)[:, 1]

    # ---- 5. Aggregate to worm-level (filename = worm)
    y_true_worm, y_pred_worm = aggregate_by_worm(
        y_test, y_pred_seg, worms_test
    )

    all_y_true.append(y_true_worm)
    all_y_pred.append(y_pred_worm)

    baseline_auc = roc_auc_score(y_true_worm, y_pred_worm)

    importances = permutation_importance_worm_level(
        pipeline,
        X_test,
        y_test,
        worms_test,
        baseline_auc,
        n_repeats=3   # keep small for speed
    )

    feature_importances_folds.append(importances)


# =========================
# Final evaluation
# =========================
y_true_all = np.concatenate(all_y_true)
y_pred_all = np.concatenate(all_y_pred)

auc = roc_auc_score(y_true_all, y_pred_all)
acc = accuracy_score(y_true_all, y_pred_all > 0.5)
bacc = balanced_accuracy_score(y_true_all, y_pred_all > 0.5)
f1 = f1_score(y_true_all, y_pred_all > 0.5)

print("\n=== Worm-level performance ===")
print(f"ROC-AUC:            {auc:.3f}")
print(f"Accuracy:           {acc:.3f}")
print(f"Balanced accuracy:  {bacc:.3f}")
print(f"F1 Score:          {f1:.3f}")

# =========================
# Feature importance analysis
# =========================
importances = np.vstack(feature_importances_folds)

mean_importance = importances.mean(axis=0)
std_importance = importances.std(axis=0)

# Sort features
order = np.argsort(mean_importance)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(
    range(len(order)),
    mean_importance[order],
    yerr=std_importance[order],
    capsize=3
)
plt.xticks(
    range(len(order)),
    np.array(feature_names)[order],
    rotation=90
)
plt.ylabel("ROC-AUC decrease (permutation importance)")
plt.title("Feature importance (mean ± std across folds)")
plt.tight_layout()
plt.show()
