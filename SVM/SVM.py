import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
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

SEGMENT_MODE  = "last"        # "first" or "last"
SEGMENT_SWEEP = list(range(1, 31))
TOP_K_FEATURES = 10

# =========================
# Helper functions
# =========================
def detect_and_drop_non_features(df, label_col, group_col):
    drop_cols = {label_col, group_col}
    for col in df.columns:
        if col in drop_cols:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            drop_cols.add(col)
        elif np.all(np.diff(df[col].values) == 1) or np.all(np.diff(df[col].values) == 0):
            drop_cols.add(col)
    return df.drop(columns=list(drop_cols))


def remove_correlated_features(X, threshold):
    corr = np.corrcoef(X, rowvar=False)
    upper = np.triu(np.abs(corr), k=1)
    to_drop = {
        j for i in range(upper.shape[0])
        for j in range(i + 1, upper.shape[1])
        if upper[i, j] > threshold
    }
    keep_idx = [i for i in range(X.shape[1]) if i not in to_drop]
    return X[:, keep_idx], keep_idx


def aggregate_by_worm_confidence_weighted(
    y_true, y_pred, worm_ids, mode="last"
):
    df = pd.DataFrame({
        "worm": worm_ids,
        "y_true": y_true,
        "y_pred": y_pred
    })

    worm_preds = []
    worm_trues = []

    for worm, g in df.groupby("worm"):
        probs = g["y_pred"].values
        true_label = g["y_true"].iloc[0]
        n = len(probs)

        if mode == "last":
            time_weights = np.linspace(0.5, 1.5, n)
        elif mode == "first":
            time_weights = np.linspace(1.5, 0.5, n)
        else:
            raise ValueError("mode must be 'first' or 'last'")

        confidence = np.abs(probs - 0.5) * 2.0
        weights = time_weights * confidence

        worm_prob = (
            np.sum(weights * probs) / np.sum(weights)
            if weights.sum() > 0 else probs.mean()
        )

        worm_preds.append(worm_prob)
        worm_trues.append(true_label)

    return np.array(worm_trues), np.array(worm_preds)


def permutation_importance_worm_level(
    model, X_test, y_test, worms_test, baseline_auc
):
    rng = np.random.default_rng(RANDOM_STATE)
    n_features = X_test.shape[1]
    importances = np.zeros(n_features)

    for j in range(n_features):
        Xp = X_test.copy()
        rng.shuffle(Xp[:, j])

        yp = model.predict_proba(Xp)[:, 1]
        yt_w, yp_w = aggregate_by_worm_confidence_weighted(
            y_test, yp, worms_test, mode=SEGMENT_MODE
        )

        importances[j] = baseline_auc - roc_auc_score(yt_w, yp_w)

    return importances


def keep_n_segments_per_worm(df, worm_col, n_segments, mode="last"):
    if mode == "last":
        return df.groupby(worm_col, group_keys=False).tail(n_segments).reset_index(drop=True)
    elif mode == "first":
        return df.groupby(worm_col, group_keys=False).head(n_segments).reset_index(drop=True)
    else:
        raise ValueError("mode must be 'first' or 'last'")


# =========================
# SVM experiment
# =========================
def run_svm_experiment(df, n_segments):
    df_sel = keep_n_segments_per_worm(
        df, worm_col=WORM_COL, n_segments=n_segments, mode=SEGMENT_MODE
    )

    worms = df_sel[WORM_COL].values
    y = df_sel[LABEL_COL].values

    df_features = detect_and_drop_non_features(df_sel, LABEL_COL, WORM_COL)
    X = df_features.values
    feature_names = df_features.columns.to_numpy()

    gkf = GroupKFold(n_splits=N_SPLITS)

    importance_folds = []
    all_y_true = []
    all_y_pred = []

    for train_idx, test_idx in gkf.split(X, y, groups=worms):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        wte = worms[test_idx]

        vt = VarianceThreshold(VAR_THRESHOLD)
        Xtr = vt.fit_transform(Xtr)
        Xte = vt.transform(Xte)
        fn = feature_names[vt.get_support()]

        Xtr, corr_idx = remove_correlated_features(Xtr, CORR_THRESHOLD)
        Xte = Xte[:, corr_idx]
        fn = fn[corr_idx]

        model = Pipeline([
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

        model.fit(Xtr, ytr)

        yp = model.predict_proba(Xte)[:, 1]
        yt_w, yp_w = aggregate_by_worm_confidence_weighted(
            yte, yp, wte, mode=SEGMENT_MODE
        )

        all_y_true.append(yt_w)
        all_y_pred.append(yp_w)

        auc0 = roc_auc_score(yt_w, yp_w)
        imp = permutation_importance_worm_level(
            model, Xte, yte, wte, auc0
        )

        importance_folds.append(pd.Series(imp, index=fn))

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    importance_mean = (
        pd.concat(importance_folds, axis=1)
          .mean(axis=1)
          .sort_values(ascending=False)
    )

    return {
        "n_segments": n_segments,
        "roc_auc": roc_auc_score(y_true_all, y_pred_all),
        "accuracy": accuracy_score(y_true_all, y_pred_all > 0.5),
        "balanced_accuracy": balanced_accuracy_score(y_true_all, y_pred_all > 0.5),
        "f1": f1_score(y_true_all, y_pred_all > 0.5),
        "top_features": importance_mean.head(TOP_K_FEATURES)
    }


# =========================
# Run sweep
# =========================
df = pd.read_csv(CSV_PATH)

results = []
feature_maps = {}

for n in SEGMENT_SWEEP:
    print(f"\nRunning SVM for N_SEGMENTS = {n}")
    res = run_svm_experiment(df, n)
    results.append(res)
    feature_maps[n] = res["top_features"]

# =========================
# Results summary
# =========================
results_df = pd.DataFrame([{
    "n_segments": r["n_segments"],
    "roc_auc": r["roc_auc"],
    "accuracy": r["accuracy"],
    "balanced_accuracy": r["balanced_accuracy"],
    "f1": r["f1"]
} for r in results])

print(results_df)

plt.figure(figsize=(7,4))
plt.plot(results_df["n_segments"], results_df["roc_auc"], marker="o", label="ROC-AUC")
plt.plot(results_df["n_segments"], results_df["f1"], marker="o", label="F1")
plt.plot(results_df["n_segments"], results_df["accuracy"], marker="o", label="Accuracy")
plt.xlabel("Number of segments per worm")
plt.ylabel("Performance")
plt.title(f"SVM performance vs segments ({SEGMENT_MODE})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
