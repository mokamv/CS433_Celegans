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

# ---------------------------------------------------------
# Load feature matrix
# ---------------------------------------------------------
df = pd.read_csv("feature_data/full_features.csv")

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
        for w in np.unique(test_worms):
            idx = test_worms == w
            worm_pred = np.mean(seg_preds[idx])   # simple average
            worm_label = y_test[idx][0]           # all segments have same label
            preds_all.append(worm_pred)
            labels_all.append(worm_label)

    # Worm-level F1 score
    preds_final = (np.array(preds_all) > 0.5).astype(int)
    return f1_score(labels_all, preds_final)


# ---------------------------------------------------------
# Run hyperparameter tuning
# ---------------------------------------------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40)

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
joblib.dump(final_model, "best_lightgbm_model.pkl")
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
plt.savefig("confusion_matrix.png")
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
plt.savefig("feature_importance_gain.png")
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
plt.savefig("feature_importance_permutation.png")
plt.close()

print("\nAll plots saved:")
print("- confusion_matrix.png")
print("- feature_importance_gain.png")
print("- feature_importance_permutation.png")
