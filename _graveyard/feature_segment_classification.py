from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from sklearn.base import clone

import numpy as np
import pandas as pd
from tqdm import tqdm
import re

from data_loader import LPBSDataLoader

import random

np.random.seed(42)
random.seed(42)


def calculate_segment_weights(weight_strategy, n_segments, segment_probs):
    """
    Assign per-segment weights according to a voting strategy.

    Supported strategies:
        - 'uniform':
            All segments have equal weight (weight 1.0).
        - 'confidence':
            Each segment is weighted by its classification confidence, i.e., the highest predicted probability among all classes for that segment.
        - 'late_segments':
            Segments are weighted linearly with increasing importance for later segments, from 0.5 (first segment) to 1.5 (last segment).
        - 'early_segments':
            Segments are weighted linearly with decreasing importance for later segments, from 1.5 (first segment) to 0.5 (last segment).
        - 'last_X_segments':
            Only the last X segments have weight 1; all earlier segments have weight 0. Replace X with an integer (e.g., 'last_10_segments').
        - 'last_X_segments_confidence':
            Only the last X segments contribute, and each is weighted by its confidence (max probability); all earlier segments are zero. Replace X with an integer (e.g., 'last_10_segments_confidence').
        - 'late_segments_confidence':
            Each segment's weight is the product of a linearly increasing value (from 0.5 to 1.5, as in 'late_segments') and its confidence (max probability per segment).

    Args:
        weight_strategy (str): Weighting strategy name (see above).
        n_segments (int): Number of segments to weight.
        segment_probs (np.ndarray): Predicted probabilities for each segment 
            (shape: [n_segments, n_classes]).

    Returns:
        np.ndarray: 1D array of weights for each segment.
    """
    if weight_strategy == 'uniform':
        weights = np.ones(n_segments)
    elif weight_strategy == 'confidence':
        weights = np.max(segment_probs, axis=1)
    elif weight_strategy == 'late_segments':
        weights = np.linspace(0.5, 1.5, n_segments)
    elif weight_strategy == 'early_segments':
        weights = np.linspace(1.5, 0.5, n_segments)
    elif re.match(r'^last_(\d+)_segments$', weight_strategy):
        match = re.match(r'^last_(\d+)_segments$', weight_strategy)
        X = int(match.group(1))
        weights = np.zeros(n_segments)
        # Assign 1 to last X segments only
        weights[-X:] = 1
    elif re.match(r'^last_(\d+)_segments_confidence$', weight_strategy):
        match = re.match(r'^last_(\d+)_segments_confidence$', weight_strategy)
        X = int(match.group(1))
        weights = np.zeros(n_segments)
        # Assign confidence to last X segments only
        weights[-X:] = np.max(segment_probs, axis=1)[-X:]
    elif weight_strategy == 'late_segments_confidence':
        weights = np.linspace(0.5, 1.5, n_segments) * np.max(segment_probs, axis=1)
    else:
        raise ValueError(f"Unsupported weight strategy: {weight_strategy}")

    return weights


def get_model(model_name: str, scaler = False) -> Pipeline:
    """Construct a scikit-learn pipeline for the requested model.

    Args:
        model_name: One of {'Limited Random Forest','Random Forest',
            'Gradient Boosting','MLP'}.
        scaler: Whether to prepend a `StandardScaler` step.

    Returns:
        sklearn.pipeline.Pipeline: Configured model pipeline ready to fit.
    """
    steps = []
    if scaler:
        steps.append(('scaler', StandardScaler()))

    # Knn and Kmeans led to bad results with all features
    if model_name == 'Limited Random Forest':
        steps.append(('classifier', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)))
    elif model_name == 'Random Forest':
        steps.append(('classifier', RandomForestClassifier(n_estimators=100, random_state=42)))
    elif model_name == 'Gradient Boosting':
        steps.append(('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42)))
    elif model_name == 'MLP':
        steps.append(('classifier', MLPClassifier(hidden_layer_sizes=(128, 64, 64), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=300, random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)))

    pipeline = Pipeline(steps)
    return pipeline


def feature_segment_classification(model: Pipeline, verbose=False, features=None):
    """Run segment-level classification with file-based CV splits.

    Args:
        model: A scikit-learn pipeline supporting `fit` and `predict_proba`.
        verbose: Whether to print dataset and metric summaries.
        features: Optional subset of feature names to select from X.

    Returns:
        dict: Mean/aggregate metrics across folds (AUC/F1/Accuracy for train/test).
    """
    loader = LPBSDataLoader()
    X, y, groups = loader.load_segment_features()
    
    if features is not None:
        X = X[features]
    
    if verbose:
        print(f"Loaded data: {X.shape[0]:,} segments, {X.shape[1]} features")
        print(f"Class distribution: {y.value_counts()}")

    cv_splits = loader.create_cv_splits(X, y, groups, n_splits=5)
    auc_scores_train = []
    auc_scores_test = []
    f1_scores_train = []
    f1_scores_test = []
    acc_scores_train = []
    acc_scores_test = []

    for fold in tqdm(cv_splits):
        X_train = fold['X_train']
        X_test = fold['X_test']
        y_train = fold['y_train']
        y_test = fold['y_test']

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        y_pred_proba = fold_model.predict_proba(X_test)[:, 1]
        y_pred_test = fold_model.predict(X_test)
        y_pred_train = fold_model.predict(X_train)

        auc_train = roc_auc_score(y_train, y_pred_train)
        auc_test = roc_auc_score(y_test, y_pred_proba)
        auc_scores_train.append(auc_train)
        auc_scores_test.append(auc_test)

        f1_train = f1_score(y_train, y_pred_train, average='binary')
        f1_test = f1_score(y_test, y_pred_test, average='binary')
        f1_scores_train.append(f1_train)
        f1_scores_test.append(f1_test)

        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        acc_scores_train.append(acc_train)
        acc_scores_test.append(acc_test)

    if verbose:
        print(f"  Mean CV Test AUC: {np.mean(auc_scores_test):.3f} ± {np.std(auc_scores_test):.3f} (Train AUC: {np.mean(auc_scores_train):.3f} ± {np.std(auc_scores_train):.3f})")
        print(f"  Mean CV Test F1: {np.mean(f1_scores_test):.3f} ± {np.std(f1_scores_test):.3f} (Train F1: {np.mean(f1_scores_train):.3f} ± {np.std(f1_scores_train):.3f})")
        print(f"  Mean CV Test Acc: {np.mean(acc_scores_test):.3f} ± {np.std(acc_scores_test):.3f} Train Acc: {np.mean(acc_scores_train):.3f} ± {np.std(acc_scores_train):.3f}")

    return {
        "mean_auc_train": np.mean(auc_scores_train),
        "mean_auc_test": np.mean(auc_scores_test),
        "mean_train_f1": np.mean(f1_scores_train),
        "mean_test_f1": np.mean(f1_scores_test),
        "mean_train_acc": np.mean(acc_scores_train),
        "mean_test_acc": np.mean(acc_scores_test)
    }


def weighted_voting_classification(model: Pipeline, weight_strategy='confidence', verbose=False, features=None):
    """Predict file-level labels via segment predictions and weighted voting.

    Args:
        model: A scikit-learn pipeline supporting `fit`, `predict`, `predict_proba`.
        weight_strategy: Strategy for weighting segments (e.g., 'uniform',
            'confidence', 'late_segments', 'last_10_segments', etc.).
        verbose: Whether to print dataset and result summaries.
        features: Optional subset of feature names to select from X.

    Returns:
        dict: File-level evaluation results including accuracy, F1, confusion
        matrix, number of files, vote analysis dataframe, and strategy name.
    """
    loader = LPBSDataLoader()
    X, y, groups = loader.load_segment_features()
    
    if features is not None:
        X = X[features]
    
    if verbose:
        print(f"Loaded: {X.shape[0]:,} segments from {groups.nunique()} files")
        print(f"Weight strategy: {weight_strategy}")

    cv_splits = loader.create_cv_splits(X, y, groups, n_splits=5)
    
    file_predictions, file_true_labels = [], []
    vote_analysis = []
    
    for fold in tqdm(cv_splits):
        fold_model = clone(model) 
        fold_model.fit(fold['X_train'], fold['y_train'])
        
        for test_file in fold['test_files']:
            file_mask = fold['groups_test'] == test_file
            file_segments = fold['X_test'][file_mask]
            file_true_label = fold['y_test'][file_mask].iloc[0]
            
            segment_preds = fold_model.predict(file_segments)
            segment_probs = fold_model.predict_proba(file_segments)
            n_segments = len(segment_preds)
            
            weights = calculate_segment_weights(weight_strategy, n_segments, segment_probs)
            
            weighted_vote_0 = np.sum(weights[segment_preds == 0])
            weighted_vote_1 = np.sum(weights[segment_preds == 1])
            file_pred = int(weighted_vote_1 > weighted_vote_0)
            
            # Calculates the confidence of the predicted file label as the proportion of the total weight that went to the winning class (the predicted class)
            total_weight = weighted_vote_0 + weighted_vote_1
            confidence = max(weighted_vote_0, weighted_vote_1) / total_weight if total_weight > 0 else 0.5
            
            vote_analysis.append({
                'n_segments': n_segments,
                'weighted_pred': file_pred,
                'weighted_confidence': confidence,
                'avg_weight': weights.mean(),
                'weight_std': weights.std(),
                'true_label': file_true_label,
                'weighted_correct': file_pred == file_true_label,
            })
            
            file_predictions.append(file_pred)
            file_true_labels.append(file_true_label)
    
    file_predictions = np.array(file_predictions)
    file_true_labels = np.array(file_true_labels)
    
    accuracy = accuracy_score(file_true_labels, file_predictions)
    f1 = f1_score(file_true_labels, file_predictions, average='binary')
    cm = confusion_matrix(file_true_labels, file_predictions)
    
    vote_df = pd.DataFrame(vote_analysis)
    
    if verbose:
        print(f"Results: {len(file_predictions)} files, Acc: {accuracy:.3f}, F1: {f1:.3f}")
        print(f"\nWeighted Voting Analysis:")
        print(f"  Weighted accuracy: {vote_df['weighted_correct'].mean():.3f}")
        print(f"  Average confidence: {vote_df['weighted_confidence'].mean():.3f}")
        
    return {
        "accuracy": accuracy,
        "f1": f1,
        "confusion_matrix": cm,
        "n_files": len(file_predictions),
        "vote_analysis": vote_df,
        "weight_strategy": weight_strategy
    }




def compare_weight_strategies(model: Pipeline, strategies=None, features=None, verbose=True):
    """Compare multiple voting strategies using file-level evaluation.

    Args:
        model: A scikit-learn pipeline.
        strategies: Iterable of strategy names (strings) to evaluate.
        features: Optional subset of features to use.
        verbose: Whether to print per-strategy summaries and ranking.

    Returns:
        dict: Mapping strategy → {accuracy, f1, avg_confidence}.
    """
    results = {}
    
    if verbose:
        print("Comparing weighted voting strategies...")
    
    for strategy in strategies:
        if verbose:
            print(f"\n--- {strategy.replace('_', ' ').title()} Weighting ---")
        
        result = weighted_voting_classification(model, strategy, verbose=False, features=features)
        vote_df = result['vote_analysis']
        
        results[strategy] = {
            'accuracy': result['accuracy'],
            'f1': result['f1'],
            'avg_confidence': vote_df['weighted_confidence'].mean(),
        }
        
        if verbose:
            print(f"  Accuracy: {results[strategy]['accuracy']:.3f}")
            print(f"  F1: {results[strategy]['f1']:.3f}")
            print(f"  Avg Confidence: {results[strategy]['avg_confidence']:.3f}")
    
    if verbose:
        print(f"\nBest Strategy Ranking:")
        sorted_strategies = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        for i, (strategy, metrics) in enumerate(sorted_strategies, 1):
            print(f"  {i}. {strategy.replace('_', ' ').title()}: {metrics['accuracy']:.3f} acc")
    
    return results



def group_prediction_cv(best_features, best_model_name, best_weight_strategy, group_size=5, n_splits=5, 
                        verbose=True):
    """Evaluate population-level classification via homogeneous groups.

    Forms groups of worms with the same true class within the CV test set and
    predicts the group label via confidence-weighted voting of individual
    worm predictions. Uses file-based CV to avoid worm leakage and applies
    class-balancing coefficients computed on the training set.

    Args:
        best_features: List of feature names to train the model on.
        best_model_name: Model name passed to `get_model`.
        best_weight_strategy: Segment weighting strategy during voting.
        group_size: Number of worms per homogeneous group.
        n_splits: Number of file-based CV splits.
        verbose: Whether to print progress and summaries.

    Returns:
        dict: Summary with accuracy, counts, per-class accuracies, and all
        group-level results across folds.
    """
    loader = LPBSDataLoader()
    X, y, groups = loader.load_segment_features()
    X = X[best_features]
    
    if verbose:
        print(f"Group Prediction with {best_model_name} + {best_weight_strategy}")
        print(f"Using {len(best_features)} features, group size: {group_size}")
        print(f"Data: {X.shape[0]:,} segments from {groups.nunique()} files")
    
    cv_splits = loader.create_cv_splits(X, y, groups, n_splits=n_splits)
    all_group_results = []
    
    for fold_idx, fold in enumerate(tqdm(cv_splits, desc="CV Folds")):
        model = get_model(best_model_name, scaler=False)
        model.fit(fold['X_train'], fold['y_train'])
        
        # Calculate class-balancing coefficients from TRAINING set to avoid data leakage
        # Get file-level predictions for training files
        train_file_predictions = {}
        for train_file in fold['train_files']:
            file_mask = fold['groups_train'] == train_file
            X_file = fold['X_train'][file_mask]
            true_label = fold['y_train'][file_mask].iloc[0]
            
            # Individual file prediction using weighted voting
            segment_preds = model.predict(X_file)
            segment_probs = model.predict_proba(X_file)
            n_segments = len(segment_preds)
            
            # Apply weighting strategy dynamically
            weights = calculate_segment_weights(best_weight_strategy, n_segments, segment_probs)
            
            weighted_vote_0 = np.sum(weights[segment_preds == 0])
            weighted_vote_1 = np.sum(weights[segment_preds == 1])
            file_pred = int(weighted_vote_1 > weighted_vote_0)
            
            train_file_predictions[train_file] = {
                'pred': file_pred, 
                'true': true_label
            }
        
        # Calculate per-class recall from TRAINING file predictions
        train_preds = np.array([train_file_predictions[f]['pred'] for f in train_file_predictions])
        train_trues = np.array([train_file_predictions[f]['true'] for f in train_file_predictions])
        
        # Class-specific recall (what proportion of each class is correctly identified)
        class_0_mask = train_trues == 0
        class_1_mask = train_trues == 1
        
        class_balancing_coef_0 = 1.0
        class_balancing_coef_1 = 1.0
        
        if class_0_mask.sum() > 0 and class_1_mask.sum() > 0:
            recall_class_0 = ((train_preds == 0) & class_0_mask).sum() / class_0_mask.sum()
            recall_class_1 = ((train_preds == 1) & class_1_mask).sum() / class_1_mask.sum()
            
            # Avoid division by zero
            recall_class_0 = max(recall_class_0, 0.01)
            recall_class_1 = max(recall_class_1, 0.01)
            
            # Balance coefficients: give higher weight to under-predicted class
            # Use inverse of recall, then normalize so geometric mean = 1
            class_balancing_coef_0 = 1.0 / recall_class_0
            class_balancing_coef_1 = 1.0 / recall_class_1
            
            # Normalize so that coef_0 * coef_1 = 1 (geometric mean = 1)
            geometric_mean = np.sqrt(class_balancing_coef_0 * class_balancing_coef_1)
            class_balancing_coef_0 /= geometric_mean
            class_balancing_coef_1 /= geometric_mean
        
        # Get file-level predictions for test files
        file_predictions = {}
        for test_file in fold['test_files']:
            file_mask = fold['groups_test'] == test_file
            X_file = fold['X_test'][file_mask]
            true_label = fold['y_test'][file_mask].iloc[0]
            
            # Individual file prediction using weighted voting
            segment_preds = model.predict(X_file)
            segment_probs = model.predict_proba(X_file)
            n_segments = len(segment_preds)
            
            # Apply weighting strategy dynamically
            weights = calculate_segment_weights(best_weight_strategy, n_segments, segment_probs)
            
            weighted_vote_0 = np.sum(weights[segment_preds == 0])
            weighted_vote_1 = np.sum(weights[segment_preds == 1])
            file_pred = int(weighted_vote_1 > weighted_vote_0)
            
            # Calculate confidence as normalized winning weight
            total_weight = weighted_vote_0 + weighted_vote_1
            confidence = max(weighted_vote_0, weighted_vote_1) / total_weight if total_weight > 0 else 0.5
            
            file_predictions[test_file] = {
                'pred': file_pred, 
                'true': true_label, 
                'confidence': confidence
            }
        
        # Group files by true class
        control_files = [f for f, data in file_predictions.items() if data['true'] == 0]
        treatment_files = [f for f, data in file_predictions.items() if data['true'] == 1]
        
        # Create groups within each class
        for class_label, files in [(0, control_files), (1, treatment_files)]:
            for i in range(0, len(files), group_size):
                group_files = files[i:i + group_size]
                if len(group_files) == group_size:  # Only complete groups
                    # Get predictions and confidences for group members
                    group_preds = [file_predictions[f]['pred'] for f in group_files]
                    group_confidences = [file_predictions[f]['confidence'] for f in group_files]
                    
                    # Confidence-weighted voting for group prediction
                    group_preds = np.array(group_preds)
                    group_confidences = np.array(group_confidences)
                    
                    weighted_vote_0 = np.sum(group_confidences[group_preds == 0])
                    weighted_vote_1 = np.sum(group_confidences[group_preds == 1])
                    
                    # Apply class-balancing coefficients
                    adjusted_vote_0 = weighted_vote_0 * class_balancing_coef_0
                    adjusted_vote_1 = weighted_vote_1 * class_balancing_coef_1
                    
                    group_prediction = int(adjusted_vote_1 > adjusted_vote_0)
                    
                    # Calculate group confidence using adjusted votes
                    total_group_weight = adjusted_vote_0 + adjusted_vote_1
                    group_confidence = max(adjusted_vote_0, adjusted_vote_1) / total_group_weight if total_group_weight > 0 else 0.5
                    
                    all_group_results.append({
                        'fold': fold_idx,
                        'true_class': class_label,
                        'predicted_class': group_prediction,
                        'individual_preds': group_preds.tolist(),
                        'individual_confidences': group_confidences.tolist(),
                        'group_confidence': group_confidence,
                        'group_size': len(group_files),
                        'class_coef_0': class_balancing_coef_0,
                        'class_coef_1': class_balancing_coef_1,
                    })
    
    # Calculate results
    correct = sum(1 for r in all_group_results if r['predicted_class'] == r['true_class'])
    total = len(all_group_results)
    accuracy = correct / total if total > 0 else 0
    
    # Class-wise results
    control_groups = [r for r in all_group_results if r['true_class'] == 0]
    treatment_groups = [r for r in all_group_results if r['true_class'] == 1]
    
    control_acc = sum(1 for r in control_groups if r['predicted_class'] == r['true_class']) / len(control_groups) if control_groups else 0
    treatment_acc = sum(1 for r in treatment_groups if r['predicted_class'] == r['true_class']) / len(treatment_groups) if treatment_groups else 0
    
    if verbose:
        avg_group_confidence = np.mean([r['group_confidence'] for r in all_group_results])
        avg_coef_0 = np.mean([r['class_coef_0'] for r in all_group_results])
        avg_coef_1 = np.mean([r['class_coef_1'] for r in all_group_results])
        
        print(f"\n=== GROUP PREDICTION RESULTS ===")
        print(f"Total groups tested: {total}")
        print(f"Overall accuracy: {accuracy:.3f} ({correct}/{total})")
        print(f"Average group confidence: {avg_group_confidence:.3f}")
        print(f"Control groups: {len(control_groups)} (accuracy: {control_acc:.3f})")
        print(f"Treatment groups: {len(treatment_groups)} (accuracy: {treatment_acc:.3f})")
        print(f"Average class coefficients: Control={avg_coef_0:.3f}, Treatment={avg_coef_1:.3f}")
        
        # Show some example predictions
        print(f"\nExample group predictions:")
        for i, result in enumerate(all_group_results[:5]):
            class_name = "Control" if result['true_class'] == 0 else "Treatment"
            pred_name = "Control" if result['predicted_class'] == 0 else "Treatment"
            correct_mark = "✓" if result['predicted_class'] == result['true_class'] else "✗"
            conf_str = f"conf:{result['group_confidence']:.3f}"
            print(f"  Group {i+1}: True={class_name}, Pred={pred_name} {correct_mark} ({conf_str}, votes: {result['individual_preds']})")
    
    return {
        'accuracy': accuracy,
        'total_groups': total,
        'correct_groups': correct,
        'control_accuracy': control_acc,
        'treatment_accuracy': treatment_acc,
        'all_results': all_group_results
    }


if __name__ == "__main__":

    # best_features = ['median_meandering_ratio', 'mean_meandering_ratio', 'min_meandering_ratio', 'wavelet_turning_level0',
    #                  'std_turning_angle', 'turning_entropy', 'wavelet_turning_level1', 'wavelet_turning_level2',
    #                  'speed_fractal_dim', 'wavelet_turning_level3']
    best_features = ['mean_meandering_ratio', 'min_meandering_ratio', 'x_iqr', 'mean_turning_angle', 'turning_entropy', 'turning_angle_mad', 'speed_acceleration_mean', 'wavelet_turning_level2', 'speed_spectral_centroid', 'turning_dominant_freq']
    best_model_name = 'Gradient Boosting'
    best_weight_strategy = 'last_30_segments_confidence'
    
    print("===== Individual Prediction =====")
    model = get_model(best_model_name, scaler=False)
    results = weighted_voting_classification(model, best_weight_strategy, verbose=False, features=best_features)
    print("Accuracy:", results['accuracy'])
    print("F1:", results['f1'])
    print("Confusion Matrix:")
    print(results['confusion_matrix'])
    
    # Calculate per-class accuracy from confusion matrix
    cm = results['confusion_matrix']
    control_acc_ind = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    treatment_acc_ind = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    print(f"Control accuracy: {control_acc_ind:.3f}, Treatment accuracy: {treatment_acc_ind:.3f}")

    print("\n===== Group Prediction =====")
    results = group_prediction_cv(best_features, best_model_name, best_weight_strategy, 
                                  group_size=5, n_splits=5, verbose=True)