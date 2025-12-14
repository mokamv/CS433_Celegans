import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------
# Load feature data
# ---------------------------------------------------------
df = pd.read_csv("feature_data/segments_features.csv") 

# Worm ID per segment
worm_ids = df["original_file"].values       # same column naming

# Segment-level labels
y = df["label"].values

# Drop non-feature columns
X = df.drop(columns=["label", "filename", "original_file"]).values

# Impute NaNs
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)
feature_names = df.drop(columns=["label", "filename", "original_file"]).columns

print("Loaded segment-level dataset:", df.shape)
print("Unique worms:", len(np.unique(worm_ids)))

# ---------------------------------------------------------
# Worm-level CV
# ---------------------------------------------------------
gkf = GroupKFold(n_splits=5)

# ---------------------------------------------------------
# Hyperparameter search (small grid that performs well)
# ---------------------------------------------------------
param_grid = {
    "n_estimators": [200, 300, 400],
    "learning_rate": [0.02, 0.03, 0.05],
    "max_depth": [2, 3],
    "subsample": [0.7, 0.8, 1.0]
}

gb = GradientBoostingClassifier()

grid = GridSearchCV(
    gb,
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

print("Running hyperparameter search...")
grid.fit(X, y)

best_params = grid.best_params_
print("\nBest parameters found:")
print(best_params)

# ---------------------------------------------------------
# Train final model
# ---------------------------------------------------------
final_model = GradientBoostingClassifier(**best_params)
final_model.fit(X, y)

joblib.dump(final_model, "gb_best_model.pkl")
print("\nSaved model as gb_best_model.pkl")

# ---------------------------------------------------------
# Worm-level CV evaluation
# ---------------------------------------------------------
worm_preds = []
worm_true = []

for train_idx, test_idx in gkf.split(X, y, groups=worm_ids):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Inside for train_idx, test_idx in gkf.split...
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    fold_model = GradientBoostingClassifier(**best_params)
    fold_model.fit(X_train, y_train)

    seg_preds = fold_model.predict_proba(X_test)[:, 1]
    test_worms = worm_ids[test_idx]

    # Worm-level aggregation (simple mean)
    for w in np.unique(test_worms):
        idx = (test_worms == w)
        worm_pred = seg_preds[idx].mean()
        worm_true_label = y_test[idx][0]
        worm_preds.append(worm_pred)
        worm_true.append(worm_true_label)

worm_preds_bin = (np.array(worm_preds) > 0.5).astype(int)

acc = accuracy_score(worm_true, worm_preds_bin)
f1 = f1_score(worm_true, worm_preds_bin)

print(f"\nWorm-level Accuracy: {acc:.4f}")
print(f"Worm-level F1 Score: {f1:.4f}")

# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------
cm = confusion_matrix(worm_true, worm_preds_bin)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Worm-level Confusion Matrix")
plt.tight_layout()
plt.savefig("gb_confusion_matrix.png")
plt.close()

print("\nSaved confusion matrix as gb_confusion_matrix.png")
