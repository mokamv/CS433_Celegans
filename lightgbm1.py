import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
# ---------------------------------------------------------
# Load feature matrix
# ---------------------------------------------------------
df = pd.read_csv("feature_data/segments_features.csv")

worm_ids = df["original_file"].values

# The label (0 = undrugged, 1 = drugged)
y = df["label"].values

# Drop non-feature columns
X = df.drop(columns=["label", "filename", "original_file"]).values
feature_names = df.drop(columns=["label", "filename", "original_file"]).columns

# ---------------------------------------------------------
# Worm-level CV without leakage
# ---------------------------------------------------------
gkf = GroupKFold(n_splits=5)


# ---------------------------------------------------------
# Aggregation helper functions accessible to `objective`
# Each function takes an array-like of segment predictions
# and returns a scalar worm-level prediction in [0,1].
# ---------------------------------------------------------

def _safe_pred(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.array([0.5])
    return arr

def agg_average(seg_preds):
    arr = _safe_pred(seg_preds)
    return float(np.mean(arr))

def agg_median(seg_preds):
    arr = _safe_pred(seg_preds)
    return float(np.median(arr))

def agg_last_k_mean(seg_preds, k=3):
    arr = _safe_pred(seg_preds)
    return float(np.mean(arr[-k:]))

def agg_weighted_linear_last_k(seg_preds, k=10):
    arr = _safe_pred(seg_preds)
    last = arr[-k:]
    n = len(last)
    weights = np.linspace(1.0, float(n), n)
    weights = weights / weights.sum()
    return float(np.sum(last * weights))

def agg_weighted_exp_recent(seg_preds, decay=0.9):
    arr = _safe_pred(seg_preds)
    n = len(arr)
    exps = decay ** (np.arange(n)[::-1])
    weights = exps / exps.sum()
    return float(np.sum(arr * weights))
def agg_trimmed_mean(seg_preds, trim=0.1):
    arr = np.asarray(seg_preds)
    if arr.size == 0:
        return 0.5
    lo = int(len(arr) * trim)
    hi = len(arr) - lo
    if hi <= lo:
        return float(arr.mean())
    return float(np.mean(np.sort(arr)[lo:hi]))

def agg_entropy_weighted(seg_preds, eps=1e-8):
    p = np.asarray(seg_preds)
    if p.size == 0:
        return 0.5
    # compute binary entropy for each prediction
    ent = -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))
    w = 1.0 / (ent + 1e-6)
    w = w / w.sum()
    return float((p * w).sum())

def agg_hybrid_percentile_or_mean(seg_preds, q=0.75, thresh=0.6):
    arr = np.asarray(seg_preds)
    if arr.size == 0:
        return 0.5
    p = float(np.percentile(arr, 100.0 * q))
    if p > thresh:
        return p
    return float(arr.mean())

AGG_METHODS = {
    'average': agg_average,
    'median': agg_median,
    'last_3_mean': lambda a: agg_last_k_mean(a, k=3),
    'last_5_mean': lambda a: agg_last_k_mean(a, k=5),
    'weighted_linear_last_10': lambda a: agg_weighted_linear_last_k(a, k=10),
    'weighted_exp_recent_decay_0.9': lambda a: agg_weighted_exp_recent(a, decay=0.9),
    'trimmed_mean': lambda a: agg_trimmed_mean(a, trim=0.1),
    'entropy_weighted': agg_entropy_weighted,
    'hybrid_percentile75_thresh0.6': lambda a: agg_hybrid_percentile_or_mean(a, q=0.75, thresh=0.6)
}

# ---------------------------------------------------------
# Objective for Optuna tuning
# ---------------------------------------------------------
def objective(trial):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 2.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 2.0)
    }

    # Collect worm-level predictions across folds
    preds_all = []
    labels_all = []
    
    for train_idx, test_idx in gkf.split(X, y, groups=worm_ids):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_data = lgb.Dataset(X_train, y_train)
        test_data  = lgb.Dataset(X_test,  y_test)

        # Use callback API for early stopping
        callbacks = [lgb.early_stopping(stopping_rounds=40, verbose=False)]

        model = lgb.train(
            params,
            train_data,
            valid_sets=[test_data],
            num_boost_round=300,
            callbacks=callbacks
        )

        # Make segment-level predictions
        seg_preds = model.predict(X_test)

        # ------------------------------
        # Aggregate worm-level predictions
        # ------------------------------
        test_worms = worm_ids[test_idx]

        # Either allow Optuna to choose aggregation method (adds a categorical hyperparameter) or set fixed
        agg_method = trial.suggest_categorical("agg_method", list(AGG_METHODS.keys()))

        for w in np.unique(test_worms):
            idx = test_worms == w
            segment_predictions = seg_preds[idx]

            if agg_method in AGG_METHODS:
                worm_pred = AGG_METHODS[agg_method](segment_predictions)
            else:
                # fallback to simple average
                worm_pred = float(np.mean(segment_predictions)) if len(segment_predictions) > 0 else 0.5

            worm_label = y_test[idx][0]  # all segments share the same worm label
            preds_all.append(worm_pred)
            labels_all.append(worm_label)

    # Worm-level F1 score
    preds_final = (np.array(preds_all) > 0.5).astype(int)
    return f1_score(labels_all, preds_final)


# ---------------------------------------------------------
# Run hyperparameter tuning
# ---------------------------------------------------------
trial_amount = 400#40
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=trial_amount)

best_params = study.best_params
print("Best params found:", best_params)

# ---------------------------------------------------------
# Final model training using full dataset
# ---------------------------------------------------------
final_model = lgb.LGBMClassifier(
    **best_params,
    n_estimators=300
)

final_model.fit(
    X, y,
    eval_set=[(X, y)],
    callbacks=[lgb.early_stopping(40, verbose=False)]
)

# Save model
joblib.dump(final_model, "results_lightgbm/best_lightgbm_model.pkl")
print("\nSaved best model as best_lightgbm_model.pkl")


# ---------------------------------------------------------
# Worm-level CV evaluation (accuracy, F1, confusion)
# ---------------------------------------------------------
worm_preds = []
worm_true = []

for train_idx, test_idx in gkf.split(X, y, groups=worm_ids):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    fold_model = lgb.LGBMClassifier(**best_params, n_estimators=300)
    fold_model.fit(X_train, y_train)

    seg_preds = fold_model.predict_proba(X_test)[:, 1]
    test_worms = worm_ids[test_idx]

    for w in np.unique(test_worms):
        idx = (test_worms == w)
        worm_pred = seg_preds[idx].mean()
        worm_preds.append(worm_pred)
        worm_true.append(y_test[idx][0])

worm_preds_bin = (np.array(worm_preds) > 0.5).astype(int)

acc = accuracy_score(worm_true, worm_preds_bin)
f1  = f1_score(worm_true, worm_preds_bin)

print(f"\nWorm-level Accuracy: {acc:.4f}")
print(f"Worm-level F1 Score: {f1:.4f}")


# ---------------------------------------------------------
# Confusion matrix (worm level)
# ---------------------------------------------------------
cm = confusion_matrix(worm_true, worm_preds_bin)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Worm-level Confusion Matrix")
plt.tight_layout()
plt.savefig("results_lightgbm/confusion_matrix.png")
plt.close()


# ---------------------------------------------------------
# Feature importance plot
# ---------------------------------------------------------
importances = final_model.booster_.feature_importance(importance_type="gain")

plt.figure(figsize=(8,10))
idx = np.argsort(importances)[::-1]
plt.barh(feature_names[idx], importances[idx])
plt.gca().invert_yaxis()
plt.title("LightGBM Feature Importance (Gain)")
plt.tight_layout()
plt.savefig("results_lightgbm/feature_importance_gain.png")
plt.close()


# ---------------------------------------------------------
# Permutation importance (slow but more reliable)
# ---------------------------------------------------------
print("\nRunning permutation importance")

perm = permutation_importance(
    final_model,
    X,
    y,
    scoring="f1",
    n_repeats=20,
    random_state=42
)

sorted_idx = perm.importances_mean.argsort()[::-1]

plt.figure(figsize=(8,10))
plt.barh(feature_names[sorted_idx], perm.importances_mean[sorted_idx])
plt.gca().invert_yaxis()
plt.title("Permutation Importance (F1 impact)")
plt.tight_layout()
plt.savefig("results_lightgbm/feature_importance_permutation.png")
plt.close()

print("\nAll plots saved:")
print("- results_lightgbm/confusion_matrix.png")
print("- results_lightgbm/feature_importance_gain.png")
print("- results_lightgbm/feature_importance_permutation.png")


# ---------------------------------------------------------
# Additional aggregation methods evaluation (worm-level)
# Uses AGG_METHODS defined above.
# ---------------------------------------------------------


# ---------------------------------------------------------
# Quick evaluation of aggregation methods using worm-level CV
# This does not modify the tuned model; it runs extra folds
# with the same GroupKFold used above and prints a comparison.
# ---------------------------------------------------------

print('\nEvaluating additional aggregation methods on worm-level CV...')
agg_results = []
for name, func in AGG_METHODS.items():
    worm_preds = []
    worm_true = []
    for train_idx, test_idx in gkf.split(X, y, groups=worm_ids):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # train a fresh fold model (lightweight LGBM with best params)
        fold_model = lgb.LGBMClassifier(**best_params, n_estimators=100)
        fold_model.fit(X_train, y_train)

        seg_preds = fold_model.predict_proba(X_test)[:, 1]
        test_worms = worm_ids[test_idx]

        for w in np.unique(test_worms):
            idx = (test_worms == w)
            worm_pred = func(seg_preds[idx])
            worm_preds.append(worm_pred)
            worm_true.append(y_test[idx][0])

    worm_preds_bin = (np.array(worm_preds) > 0.5).astype(int)
    acc = accuracy_score(worm_true, worm_preds_bin)
    f1 = f1_score(worm_true, worm_preds_bin)
    agg_results.append((name, acc, f1))
    print(f"{name}: Accuracy={acc:.4f}, F1={f1:.4f}")

# Optionally save results and plot comparison
os.makedirs('results_lightgbm', exist_ok=True)
res_df = pd.DataFrame(agg_results, columns=['agg_method', 'accuracy', 'f1'])

# Print highest F1 and highest accuracy results (include the other metric too)
best_f1_row = res_df.loc[res_df['f1'].idxmax()]
best_acc_row = res_df.loc[res_df['accuracy'].idxmax()]
print(f"Highest F1: {best_f1_row['f1']:.4f} (method: {best_f1_row['agg_method']}), Accuracy: {best_f1_row['accuracy']:.4f}")
print(f"Highest Accuracy: {best_acc_row['accuracy']:.4f} (method: {best_acc_row['agg_method']}), F1: {best_acc_row['f1']:.4f}")

# Plot accuracy and F1 for each aggregation method (sorted by F1)
res_df_sorted = res_df.sort_values('f1', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
sns.barplot(x='accuracy', y='agg_method', data=res_df_sorted, ax=axes[0], palette='Blues_d')
axes[0].set_title('Aggregation Methods — Accuracy')
axes[0].set_xlabel('Accuracy')

sns.barplot(x='f1', y='agg_method', data=res_df_sorted, ax=axes[1], palette='Greens_d')
axes[1].set_title('Aggregation Methods — F1 Score')
axes[1].set_xlabel('F1 Score')

plt.tight_layout()
plt.savefig('results_lightgbm/agg_method_comparison.png')
plt.close()
print('Saved plot to results_lightgbm/agg_method_comparison.png')