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
from sklearn.metrics import confusion_matrix
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
SEGMENT_MODE = "last"   # options: "last", "first"


SEGMENT_SWEEP = list(range(1, 31))

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


def keep_n_segments_per_worm(df, worm_col, n_segments, mode="last"):
    """
    Keep only the first or last n_segments per worm.
    """
    if n_segments is None:
        return df

    if mode == "last":
        return (
            df.groupby(worm_col, group_keys=False)
              .tail(n_segments)
              .reset_index(drop=True)
        )
    elif mode == "first":
        return (
            df.groupby(worm_col, group_keys=False)
              .head(n_segments)
              .reset_index(drop=True)
        )
    else:
        raise ValueError("mode must be 'first' or 'last'")



def run_logreg_experiment(df, n_segments):
    df_sel = keep_n_segments_per_worm(
        df,
        worm_col=WORM_COL,
        n_segments=n_segments,
        mode=SEGMENT_MODE
    )

    # Prevent data leakage between worms. GroupKFold on worm IDs.
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
            mode=SEGMENT_MODE
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
    cm = confusion_matrix(y_true_all, preds)

    return {
        "n_segments": n_segments,
        "best_threshold": best_t,
        "roc_auc": roc_auc_score(y_true_all, y_pred_all),
        "f1": f1_score(y_true_all, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true_all, preds),
        "accuracy": accuracy_score(y_true_all, preds),
        "confusion_matrix": cm
    }

def aggregate_by_worm_confidence_weighted(
    y_true,
    y_pred,
    worm_ids,
    mode="last"
):
    """
    Aggregate segment-level probabilities to worm-level
    using confidence-weighted + time-weighted voting.

    If mode == 'last': later segments get higher weight.
    If mode == 'first': earlier segments get higher weight.
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

        if mode == "last":
            time_weights = np.linspace(0.5, 1.5, n)
        elif mode == "first":
            time_weights = np.linspace(1.5, 0.5, n)
        else:
            raise ValueError("mode must be 'first' or 'last'")

        confidence = np.abs(probs - 0.5) * 2.0
        weights = time_weights * confidence

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
    print(f"\nRunning for N_SEGMENTS = {n}")
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

x = results_df["n_segments"].fillna(
    results_df["n_segments"].max() + 5
)

plt.plot(x, results_df["roc_auc"], marker="o", label="ROC-AUC")
plt.plot(x, results_df["f1"], marker="o", label="F1 score")
plt.plot(x, results_df["accuracy"], marker="o", label="Accuracy")

plt.xlabel("Number of segments used")
plt.ylabel("Performance")
plt.title("Performance vs number of segments used")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# Feature importance for best accuracy case
# =========================

# 1. Identify best-performing configuration
best_row = results_df.loc[results_df["accuracy"].idxmax()]
best_n = int(best_row["n_segments"])

print("\n=== Best configuration ===")
print(best_row)
print(f"Using N_SEGMENTS = {best_n}, mode = {SEGMENT_MODE}")

# 2. Rebuild dataset for best case
df_best = keep_n_segments_per_worm(
    df,
    worm_col=WORM_COL,
    n_segments=best_n,
    mode=SEGMENT_MODE
)

y = df_best[LABEL_COL].values

df_features = detect_and_drop_non_features(
    df_best, LABEL_COL, WORM_COL
)

X = df_features.values
feature_names = df_features.columns.to_numpy()

# 3. Apply same feature filtering as during CV
vt = VarianceThreshold(VAR_THRESHOLD)
X = vt.fit_transform(X)
feature_names = feature_names[vt.get_support()]

X, corr_idx = remove_correlated_features(X, CORR_THRESHOLD)
feature_names = feature_names[corr_idx]

print(f"Number of features after filtering: {len(feature_names)}")

# 4. Train Logistic Regression once (diagnostic model)
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

pipeline.fit(X, y)

# 5. Extract coefficients as feature importance
coef = pipeline.named_steps["logreg"].coef_[0]
importance = np.abs(coef)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coef,
    "importance": importance
}).sort_values("importance", ascending=False)

print("\nTop important features:")
print(importance_df.head(10))

# 6. Plot top-K features
TOP_K = 15

plt.figure(figsize=(8, 5))
plt.barh(
    importance_df["feature"].iloc[:TOP_K][::-1],
    importance_df["importance"].iloc[:TOP_K][::-1]
)
plt.xlabel("Absolute coefficient magnitude")
plt.title(
    f"Top {TOP_K} feature importances\n"
    f"(Logistic Regression, N_SEGMENTS={best_n}, mode={SEGMENT_MODE})"
)
plt.tight_layout()
plt.show()

# =========================
# Confusion matrix for best accuracy case
# =========================

best_result = max(results, key=lambda r: r["accuracy"])

print("\n=== Best configuration ===")
print(f"N_SEGMENTS = {best_result['n_segments']}")
print(f"Accuracy   = {best_result['accuracy']:.3f}")
print(f"F1 score   = {best_result['f1']:.3f}")
print(f"Threshold  = {best_result['best_threshold']:.2f}")

cm = best_result["confusion_matrix"]

print("\nConfusion matrix (worm-level):")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nDerived metrics:")
print(f"Recall (control):  {tn / (tn + fp + 1e-6):.3f}")
print(f"Recall (treated):  {tp / (tp + fn + 1e-6):.3f}")
print(f"Precision (treated): {tp / (tp + fp + 1e-6):.3f}")
