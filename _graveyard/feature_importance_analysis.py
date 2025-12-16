"""
AI Generated

Feature Importance Analysis for LPBS Dataset
Find the most important features for prediction and test performance with feature subsets.
"""

import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from data_loader import LPBSDataLoader

# Output directories
FIGURES_DIR = "Figures"
RESULTS_DIR = "Results"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class FeatureImportanceAnalyzer:
    def __init__(self):
        self.loader = LPBSDataLoader()

        # ---- X IS LOADED HERE ----
        self.X, self.y, self.groups = self.loader.load_segment_features()

        # ---- REMOVE HIGHLY CORRELATED FEATURES ----
        self.remove_highly_correlated_features(threshold=0.95)

        # ---- IMMEDIATE NaN DIAGNOSTIC (earliest possible) ----
        nan_counts = self.X.isna().sum()
        nan_features = nan_counts[nan_counts > 0]

        if len(nan_features) > 0:
            print("\n[WARNING] NaNs detected in feature matrix X")
            print("Affected features and NaN counts:")
            print(nan_features.sort_values(ascending=False))
        else:
            print("\n[OK] No NaNs detected in feature matrix X")

        # ---- HARDEN X FOR ALL DOWNSTREAM MODELS ----
        self.X = self.X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Metadata
        self.feature_names = self.X.columns.tolist()
        print(f"Loaded {len(self.X)} samples with {len(self.feature_names)} features")

    def remove_highly_correlated_features(self, threshold=0.95):
        """Remove features with a correlation higher than the threshold (keep only one from each group)."""
        print(f"Checking for highly correlated features (threshold={threshold})...")
        corr_matrix = self.X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        if to_drop:
            print(f"Removing {len(to_drop)} highly correlated features: {to_drop}")
        else:
            print("No highly correlated features found.")
        self.X = self.X.drop(columns=to_drop)
        
    def random_forest_importance(self, n_estimators=100):
        """Calculate feature importance using Random Forest"""
        print("Calculating Random Forest feature importance...")
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf.fit(self.X, self.y)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'rf_importance': rf.feature_importances_
        }).sort_values('rf_importance', ascending=False)
        
        return importance_df
    
    def permutation_importance_analysis(self, n_repeats=10):
        """Calculate permutation importance"""
        print("Calculating permutation importance...")
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(self.X, self.y)
        
        perm_importance = permutation_importance(
            rf, self.X, self.y, n_repeats=n_repeats, random_state=42, n_jobs=-1
        )
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'perm_importance': perm_importance.importances_mean,
            'perm_std': perm_importance.importances_std
        }).sort_values('perm_importance', ascending=False)
        
        return importance_df
    
    def statistical_importance(self):
        """Calculate statistical importance using F-score and mutual information"""
        print("Calculating statistical importance...")
        
        # F-score
        f_scores, _ = f_classif(self.X, self.y)
        
        # Mutual Information
        mi_scores = mutual_info_classif(self.X, self.y, random_state=42)
        
        # Correlation with target
        correlations = []
        for col in self.feature_names:
            corr = abs(np.corrcoef(self.X[col], self.y)[0, 1])
            correlations.append(corr)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'f_score': f_scores,
            'mutual_info': mi_scores,
            'abs_correlation': correlations
        })
        
        return importance_df
    
    def combine_importance_scores(self, rf_df, perm_df, stat_df):
        """Combine different importance metrics"""
        print("Combining importance scores...")
        
        # Merge all dataframes
        combined = rf_df[['feature', 'rf_importance']].copy()
        combined = combined.merge(perm_df[['feature', 'perm_importance']], on='feature')
        combined = combined.merge(stat_df, on='feature')
        
        # Normalize scores to 0-1 range
        for col in ['rf_importance', 'perm_importance', 'f_score', 'mutual_info', 'abs_correlation']:
            combined[f'{col}_norm'] = (combined[col] - combined[col].min()) / (combined[col].max() - combined[col].min())
        
        # Calculate combined score (weighted average)
        combined['combined_score'] = (
            0.3 * combined['rf_importance_norm'] +
            0.3 * combined['perm_importance_norm'] +
            0.2 * combined['f_score_norm'] +
            0.1 * combined['mutual_info_norm'] +
            0.1 * combined['abs_correlation_norm']
        )
        
        return combined.sort_values('combined_score', ascending=False)
    
    def test_feature_subset_performance(self, importance_df, max_features=50):
        """Test performance with different numbers of top features"""
        print(f"Testing performance with 1 to {max_features} top features...")
        
        feature_counts = range(1, min(max_features + 1, len(self.feature_names) + 1), 2)
        results = []
        
        for n_features in tqdm(feature_counts, desc="Testing feature subsets"):
            # Select top n features
            top_features = importance_df.head(n_features)['feature'].tolist()
            X_subset = self.X[top_features]
            
            # Train Random Forest with cross-validation
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            cv_scores = cross_val_score(rf, X_subset, self.y, cv=5, scoring='roc_auc')
            
            results.append({
                'n_features': n_features,
                'mean_auc': cv_scores.mean(),
                'std_auc': cv_scores.std()
            })
        
        return pd.DataFrame(results)
    
    def plot_importance(self, importance_df, top_n=20):
        """Plot feature importance"""
        plt.figure(figsize=(12, 8))
        
        # Plot top features
        top_features = importance_df.head(top_n)
        
        plt.subplot(2, 2, 1)
        plt.barh(range(len(top_features)), top_features['rf_importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Random Forest Importance')
        plt.title('Top Features by RF Importance')
        plt.gca().invert_yaxis()
        
        plt.subplot(2, 2, 2)
        plt.barh(range(len(top_features)), top_features['perm_importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Permutation Importance')
        plt.title('Top Features by Permutation Importance')
        plt.gca().invert_yaxis()
        
        plt.subplot(2, 2, 3)
        plt.barh(range(len(top_features)), top_features['combined_score'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Combined Score')
        plt.title('Top Features by Combined Score')
        plt.gca().invert_yaxis()
        
        plt.subplot(2, 2, 4)
        plt.scatter(top_features['rf_importance'], top_features['perm_importance'])
        plt.xlabel('Random Forest Importance')
        plt.ylabel('Permutation Importance')
        plt.title('RF vs Permutation Importance')
        
        for i, feature in enumerate(top_features['feature']):
            if i < 10:  # Label only top 10 to avoid clutter
                plt.annotate(feature[:15], 
                           (top_features.iloc[i]['rf_importance'], 
                            top_features.iloc[i]['perm_importance']),
                           fontsize=8)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(FIGURES_DIR, "feature_importance_analysis.png"),
            dpi=300,
            bbox_inches='tight'
        )

        plt.show()
    
    def plot_performance_vs_features(self, performance_df):
        """Plot performance vs number of features"""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(performance_df['n_features'], performance_df['mean_auc'], 'b-', alpha=0.7)
        plt.fill_between(performance_df['n_features'], 
                        performance_df['mean_auc'] - performance_df['std_auc'],
                        performance_df['mean_auc'] + performance_df['std_auc'], 
                        alpha=0.3)
        plt.xlabel('Number of Features')
        plt.ylabel('Cross-Validation AUC')
        plt.title('Performance vs Number of Features')
        plt.grid(True, alpha=0.3)
        
        # Find optimal number of features
        optimal_idx = performance_df['mean_auc'].idxmax()
        optimal_features = performance_df.loc[optimal_idx, 'n_features']
        optimal_auc = performance_df.loc[optimal_idx, 'mean_auc']
        
        plt.axvline(x=optimal_features, color='red', linestyle='--', alpha=0.7)
        plt.text(optimal_features + 1, optimal_auc, 
                f'Optimal: {optimal_features} features\nAUC: {optimal_auc:.3f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.subplot(1, 2, 2)
        # Performance improvement over baseline
        baseline_auc = performance_df['mean_auc'].iloc[0]  # Single feature performance
        improvement = performance_df['mean_auc'] - baseline_auc
        
        plt.plot(performance_df['n_features'], improvement, 'g-', alpha=0.7)
        plt.xlabel('Number of Features')
        plt.ylabel('AUC Improvement over Single Feature')
        plt.title('Performance Improvement')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(FIGURES_DIR, "performance_vs_features.png"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()
        
        return optimal_features, optimal_auc
    
    def save_top_features(self, importance_df, top_n=20):
        """Save top features to CSV for easy access"""
        top_features = importance_df.head(top_n)
        top_features.to_csv(
            os.path.join(RESULTS_DIR, "top_features.csv"),
            index=False
        )
        print(f"\nTop {top_n} features saved to 'top_features.csv'")
        return top_features['feature'].tolist()
    
    def run_full_analysis(self):
        """Run the complete feature importance analysis"""
        print("Starting comprehensive feature importance analysis...")
        print("=" * 60)
        
        # Calculate different importance metrics
        rf_importance = self.random_forest_importance()
        perm_importance = self.permutation_importance_analysis()
        stat_importance = self.statistical_importance()
        
        # Combine all metrics
        combined_importance = self.combine_importance_scores(rf_importance, perm_importance, stat_importance)
        
        # Test performance with different feature subsets
        performance_results = self.test_feature_subset_performance(combined_importance)
        
        # Create visualizations
        self.plot_importance(combined_importance)
        optimal_features, optimal_auc = self.plot_performance_vs_features(performance_results)
        
        # Save results
        top_features_list = self.save_top_features(combined_importance, top_n=20)
        combined_importance.to_csv(
            os.path.join(RESULTS_DIR, "all_feature_importance.csv"),
            index=False
        )
        performance_results.to_csv(
            os.path.join(RESULTS_DIR, "feature_performance_analysis.csv"),
            index=False
        )

        
        # Print summary
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total features analyzed: {len(self.feature_names)}")
        print(f"Optimal number of features: {optimal_features}")
        print(f"Best AUC achieved: {optimal_auc:.3f}")
        print(f"Performance with all features: {performance_results['mean_auc'].iloc[-1]:.3f}")
        
        print(f"\nTop 10 most important features:")
        for i, (_, row) in enumerate(combined_importance.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature'][:40]} (score: {row['combined_score']:.3f})")

        print(combined_importance)
        
        return {
            'importance_df': combined_importance,
            'performance_df': performance_results,
            'optimal_n_features': optimal_features,
            'optimal_auc': optimal_auc,
            'top_features': top_features_list
        }


def quick_feature_test(n_features=10):
    """Quick test with top n features"""
    analyzer = FeatureImportanceAnalyzer()
    
    # Get feature importance quickly
    rf_importance = analyzer.random_forest_importance(n_estimators=50)
    top_features = rf_importance.head(n_features)['feature'].tolist()
    
    # Test performance
    X_subset = analyzer.X[top_features]
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(rf, X_subset, analyzer.y, cv=5, scoring='roc_auc')
    
    print(f"\nQuick Test Results:")
    print(f"Using top {n_features} features: {top_features}")
    print(f"Cross-validation AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    return top_features, cv_scores.mean()


if __name__ == "__main__":
    # Run full analysis
    analyzer = FeatureImportanceAnalyzer()
    results = analyzer.run_full_analysis()
    
    # Quick test with different numbers of features
    print("\n" + "=" * 60)
    print("QUICK PERFORMANCE TESTS")
    print("=" * 60)
    
    for n in [5, 10, 15, 20]:
        top_features, auc = quick_feature_test(n)
        print(f"Top {n:2d} features AUC: {auc:.3f}")