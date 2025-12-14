import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)
from sklearn.feature_selection import VarianceThreshold

# =========================
# Config
# =========================
CSV_PATH = "feature_data/segments_features.csv"

WORM_COL  = "original_file"
LABEL_COL = "label"

N_SPLITS = 5
CORR_THRESHOLD = 0.95
VAR_THRESHOLD = 1e-4

RANDOM_STATE = 42

SEGMENT_SWEEP = list(range(1, 41))

# =========================
# Helper functions
# =========================
def detect_and_drop_non_features(df, label_col, group_col):
    drop_cols = set([label_col, group_col])

    for col in df.columns:
        if col in drop_cols:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            drop_cols.add(col)

    for col in df.columns:
        if col in drop_cols:
            continue
        values = df[col].values
        if np.all(np.diff(values) == 1) or np.all(np.diff(values) == 0):
            drop_cols.add(col)

    return df.drop(columns=list(drop_cols))


def remove_correlated_features(X, threshold):
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
    df = pd.DataFrame({
        "worm": worm_ids,
        "y_true": y_true,
        "y_pred": y_pred
    })

    grouped = df.groupby("worm").mean()
    return grouped["y_true"].values, grouped["y_pred"].values


def keep_last_n_segments_per_worm(df, worm_col, n_segments):
    if n_segments is None:
        return df
    return (
        df.groupby(worm_col, group_keys=False)
          .tail(n_segments)
          .reset_index(drop=True)
    )


def run_logreg_experiment(df, last_n_segments):
    df_sel = keep_last_n_segments_per_worm(
        df, worm_col=WORM_COL, n_segments=last_n_segments
    )

    worms = df_sel[WORM_COL].values
    y = df_sel[LABEL_COL].values

    df_features = detect_and_drop_non_features(
        df_sel, LABEL_COL, WORM_COL
    )
    X = df_features.values

    gkf = GroupKFold(n_splits=N_SPLITS)

    all_y_true = []
    all_y_pred = []

    for train_idx, test_idx in gkf.split(X, y, groups=worms):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        worms_test = worms[test_idx]

        vt = VarianceThreshold(VAR_THRESHOLD)
        X_train = vt.fit_transform(X_train)
        X_test = vt.transform(X_test)

        X_train, corr_idx = remove_correlated_features(
            X_train, CORR_THRESHOLD
        )
        X_test = X_test[:, corr_idx]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE
            ))
        ])

        pipeline.fit(X_train, y_train)
        y_pred_seg = pipeline.predict_proba(X_test)[:, 1]

        y_true_w, y_pred_w = aggregate_by_worm_confidence_weighted(
            y_test,
            y_pred_seg,
            worms_test,
            last_n_segments
        )


        all_y_true.append(y_true_w)
        all_y_pred.append(y_pred_w)

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    thresholds = np.linspace(0.05, 0.95, 50)
    f1s = [f1_score(y_true_all, y_pred_all > t) for t in thresholds]

    best_idx = np.argmax(f1s)
    best_t = thresholds[best_idx]

    preds = y_pred_all > best_t

    return {
        "last_n_segments": last_n_segments,
        "best_threshold": best_t,
        "roc_auc": roc_auc_score(y_true_all, y_pred_all),
        "f1": f1_score(y_true_all, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true_all, preds),
        "accuracy": accuracy_score(y_true_all, preds),
    }

def aggregate_by_worm_confidence_weighted(
    y_true,
    y_pred,
    worm_ids,
    n_segments_per_worm
):
    """
    Aggregate segment-level probabilities to worm-level
    using confidence-weighted + time-weighted voting.

    Later segments get higher weight than earlier ones.
    """
    df = pd.DataFrame({
        "worm": worm_ids,
        "y_true": y_true,
        "y_pred": y_pred
    })

    worm_preds = []
    worm_trues = []

    for worm, group in df.groupby("worm"):
        probs = group["y_pred"].values
        true_label = group["y_true"].iloc[0]

        n = len(probs)

        # --- time weights: linearly increasing
        time_weights = np.linspace(0.5, 1.5, n)

        # --- confidence = distance from 0.5
        confidence = np.abs(probs - 0.5) * 2.0

        weights = time_weights * confidence

        # avoid zero division
        if weights.sum() == 0:
            worm_prob = probs.mean()
        else:
            worm_prob = np.sum(weights * probs) / np.sum(weights)

        worm_preds.append(worm_prob)
        worm_trues.append(true_label)

    return np.array(worm_trues), np.array(worm_preds)


# =========================
# Run sweep
# =========================
df = pd.read_csv(CSV_PATH)

results = []

for n in SEGMENT_SWEEP:
    print(f"\nRunning for LAST_N_SEGMENTS = {n}")
    res = run_logreg_experiment(df, n)
    print(res)
    results.append(res)

results_df = pd.DataFrame(results)
print("\nSummary:")
print(results_df)

# =========================
# Plot performance vs segments
# =========================
plt.figure(figsize=(7, 4))

x = results_df["last_n_segments"].fillna(
    results_df["last_n_segments"].max() + 5
)

plt.plot(x, results_df["roc_auc"], marker="o", label="ROC-AUC")
plt.plot(x, results_df["f1"], marker="o", label="F1 score")
plt.plot(x, results_df["balanced_accuracy"], marker="o", label="Balanced accuracy")

plt.xlabel("Number of last segments used")
plt.ylabel("Performance")
plt.title("Performance vs number of last segments")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
