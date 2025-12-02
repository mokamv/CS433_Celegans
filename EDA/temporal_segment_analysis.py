import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import savgol_filter
import os
import argparse
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

def setup_plotting_style():
    """Set up the style for all plots"""
    plt.style.use('ggplot')
    sns.set_theme(font_scale=1.1)
    sns.set_style("whitegrid")
    plt.rcParams['figure.max_open_warning'] = 50

def load_data(data_type):
    """Load the data based on the data type."""
    if data_type == 'full':
        data_file = 'feature_data/full_features.csv'
        plot_dir = 'EDA/full_data/temporal_analysis'
    else:
        data_file = 'feature_data/segments_features.csv'
        plot_dir = 'EDA/segments_data/temporal_analysis'
    
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Loading {data_type} data from {data_file}...")
    df = pd.read_csv(data_file)
    return df, plot_dir

def analyze_temporal_trends(df, plot_dir):
    """Analyze how features change with relative lifespan percentage."""
    print("\n=== Temporal Analysis: Features vs Relative Lifespan Percentage ===")
    
    # Only analyze segments data as full data doesn't have segment_index
    if 'segment_index' not in df.columns:
        print("No segment_index column found. This analysis is only for segments data.")
        return
    
    # Calculate relative lifespan percentage for each segment
    print("Calculating relative lifespan percentages...")
    df = calculate_relative_lifespan_percentage(df)
    
    # Key features to analyze for temporal patterns
    temporal_features = [
        'mean_speed', 'std_speed', 'max_speed',
        'total_distance', 'time_paused', 'fraction_paused',
        'mean_turning_angle', 'std_turning_angle', 'turning_frequency',
        'mean_frenetic_score', 'max_frenetic_score', 'std_frenetic_score', 'pct_high_frenetic',
        'mean_roaming_score', 'std_roaming_score', 'fraction_roaming',
        'activity_level', 'high_activity_fraction', 'low_activity_fraction',
        'movement_efficiency', 'speed_persistence', 'exploration_ratio',
        'kinetic_energy_proxy', 'mean_jerk', 'max_jerk'
    ]
    
    # Filter features that exist in the data
    available_features = [f for f in temporal_features if f in df.columns]
    print(f"Analyzing {len(available_features)} features across relative lifespan percentages")
    
    # Create comprehensive temporal analysis
    results = {}
    
    # 0. Analyze rotation effects
    print("\n0. Analyzing rotation augmentation effects...")
    rotation_results = analyze_rotation_effects(df, available_features, plot_dir)
    results['rotation_effects'] = rotation_results
    
    # 1. Overall temporal trends
    print("\n1. Analyzing overall temporal trends...")
    temporal_stats = analyze_overall_temporal_trends(df, available_features)
    results['overall_trends'] = temporal_stats
    
    # 2. Condition-specific temporal trends
    print("2. Analyzing condition-specific temporal trends...")
    condition_stats = analyze_condition_temporal_trends(df, available_features)
    results['condition_trends'] = condition_stats
    
    # 3. Visualizations
    print("3. Creating visualizations...")
    create_temporal_visualizations(df, available_features, plot_dir)
    
    # 4. Statistical analysis
    print("4. Performing statistical analysis...")
    statistical_results = perform_temporal_statistical_analysis(df, available_features)
    results['statistical_analysis'] = statistical_results
    
    # 5. First vs Last 50% analysis
    print("5. Analyzing first 50% vs last 50% of life...")
    first_vs_last_results = analyze_first_vs_last_50_percent(df, available_features, plot_dir)
    results['first_vs_last_50_percent'] = first_vs_last_results
    
    # Save results
    save_temporal_analysis_results(results, plot_dir)
    
    print(f"\nTemporal analysis complete! Results saved to {plot_dir}")

def calculate_relative_lifespan_percentage(df):
    """Calculate relative lifespan percentage for each segment within each worm."""
    # Group by original_file to get total segments per worm
    worm_stats = df.groupby('original_file').agg({
        'segment_index': ['max', 'count']
    }).reset_index()
    worm_stats.columns = ['original_file', 'max_segment', 'total_segments']
    
    # Merge back to original dataframe
    df = df.merge(worm_stats, on='original_file', how='left')
    
    # Calculate relative lifespan percentage (0% = start of life, 100% = end of life)
    df['lifespan_percentage'] = (df['segment_index'] / df['max_segment']) * 100
    
    # Also create bins for analysis
    df['lifespan_bin'] = pd.cut(df['lifespan_percentage'], 
                               bins=[0, 20, 40, 60, 80, 100], 
                               labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
    
    # Analyze rotation augmentation
    rotation_stats = df.groupby(['original_file', 'segment_index']).size().reset_index(name='rotation_count')
    avg_rotations = rotation_stats['rotation_count'].mean()
    unique_segments = len(rotation_stats)
    total_entries = len(df)
    
    print(f"Lifespan percentage range: {df['lifespan_percentage'].min():.1f}% - {df['lifespan_percentage'].max():.1f}%")
    print(f"Average segments per worm: {df['total_segments'].mean():.1f}")
    print(f"Number of unique worms: {df['original_file'].nunique()}")
    print(f"Rotation augmentation: {avg_rotations:.1f} rotations per segment")
    print(f"Total segments (without rotations): {unique_segments}")
    print(f"Total entries (with rotations): {total_entries}")
    
    return df

def analyze_rotation_effects(df, features, plot_dir):
    """Analyze how rotation augmentation affects feature distributions."""
    print("\n=== Rotation Augmentation Analysis ===")
    
    # Group by segment to analyze rotation effects
    rotation_analysis = {}
    
    for feature in features[:10]:  # Analyze first 10 features to avoid too much output
        if feature in df.columns:
            # Calculate coefficient of variation (std/mean) for each segment's rotations
            segment_stats = df.groupby(['original_file', 'segment_index'])[feature].agg(['mean', 'std', 'count']).reset_index()
            segment_stats = segment_stats[segment_stats['count'] > 1]  # Only segments with multiple rotations
            
            if len(segment_stats) > 0:
                # Calculate coefficient of variation
                segment_stats['cv'] = segment_stats['std'] / segment_stats['mean'].abs()
                segment_stats = segment_stats[segment_stats['mean'] != 0]  # Remove zero means
                
                if len(segment_stats) > 0:
                    rotation_analysis[feature] = {
                        'mean_cv': segment_stats['cv'].mean(),
                        'std_cv': segment_stats['cv'].std(),
                        'max_cv': segment_stats['cv'].max(),
                        'min_cv': segment_stats['cv'].min(),
                        'segments_analyzed': len(segment_stats)
                    }
    
    # Save rotation analysis
    if rotation_analysis:
        rotation_df = pd.DataFrame(rotation_analysis).T
        rotation_df.to_csv(os.path.join(plot_dir, 'rotation_effects_analysis.csv'))
        
        print(f"Rotation effects analyzed for {len(rotation_analysis)} features")
        print("Top 5 features with highest rotation variability:")
        sorted_features = sorted(rotation_analysis.items(), key=lambda x: x[1]['mean_cv'], reverse=True)
        for feature, stats in sorted_features[:5]:
            print(f"  - {feature}: CV = {stats['mean_cv']:.3f} ± {stats['std_cv']:.3f}")
    
    return rotation_analysis

def analyze_overall_temporal_trends(df, features):
    """Analyze overall trends across relative lifespan percentages."""
    results = {}
    
    # Group by lifespan percentage bins and calculate means
    lifespan_means = df.groupby('lifespan_bin')[features].mean()
    
    # Calculate trends for each feature
    for feature in features:
        lifespan_percentages = df.groupby('lifespan_bin')[feature].mean().index
        feature_values = lifespan_means[feature].values
        
        # Remove NaN values
        valid_mask = ~np.isnan(feature_values)
        if valid_mask.sum() < 3:
            continue
            
        clean_percentages = np.array([float(str(x).split('-')[0]) for x in lifespan_percentages[valid_mask]])
        clean_values = feature_values[valid_mask]
        
        # Linear regression to find trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(clean_percentages, clean_values)
        
        # Calculate percentage change from early to late life
        first_val = clean_values[0] if len(clean_values) > 0 else np.nan
        last_val = clean_values[-1] if len(clean_values) > 0 else np.nan
        pct_change = ((last_val - first_val) / abs(first_val)) * 100 if first_val != 0 else np.nan
        
        results[feature] = {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'significance': 'significant' if p_value < 0.05 else 'not_significant',
            'trend_direction': 'increasing' if slope > 0 else 'decreasing',
            'percentage_change': pct_change,
            'early_life_mean': first_val,
            'late_life_mean': last_val
        }
    
    return results

def analyze_condition_temporal_trends(df, features):
    """Analyze temporal trends separately for each condition."""
    results = {}
    
    for condition in df['label'].unique():
        condition_df = df[df['label'] == condition]
        condition_results = {}
        
        # Group by lifespan percentage bins and calculate means
        lifespan_means = condition_df.groupby('lifespan_bin')[features].mean()
        
        for feature in features:
            lifespan_percentages = condition_df.groupby('lifespan_bin')[feature].mean().index
            feature_values = lifespan_means[feature].values
            
            # Remove NaN values
            valid_mask = ~np.isnan(feature_values)
            if valid_mask.sum() < 3:
                continue
                
            clean_percentages = np.array([float(str(x).split('-')[0]) for x in lifespan_percentages[valid_mask]])
            clean_values = feature_values[valid_mask]
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(clean_percentages, clean_values)
            
            # Calculate percentage change
            first_val = clean_values[0] if len(clean_values) > 0 else np.nan
            last_val = clean_values[-1] if len(clean_values) > 0 else np.nan
            pct_change = ((last_val - first_val) / abs(first_val)) * 100 if first_val != 0 else np.nan
            
            condition_results[feature] = {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'significance': 'significant' if p_value < 0.05 else 'not_significant',
                'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                'percentage_change': pct_change
            }
        
        results[f'condition_{condition}'] = condition_results
    
    return results

def create_temporal_visualizations(df, features, plot_dir):
    """Create comprehensive visualizations of temporal patterns."""
    
    # 1. Multi-panel plot for key behavioral features
    key_features = ['mean_speed', 'fraction_paused', 'mean_frenetic_score', 
                   'fraction_roaming', 'activity_level', 'movement_efficiency']
    available_key_features = [f for f in key_features if f in features]
    
    if len(available_key_features) >= 4:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(available_key_features[:6]):
            ax = axes[i]
            
            # Plot for each condition
            for condition in df['label'].unique():
                condition_df = df[df['label'] == condition]
                lifespan_stats = condition_df.groupby('lifespan_bin')[feature].agg(['mean', 'sem']).reset_index()
                
                # Convert bin labels to numeric for plotting
                bin_centers = [10, 30, 50, 70, 90]  # Center of each 20% bin
                ax.plot(bin_centers[:len(lifespan_stats)], lifespan_stats['mean'], 
                       marker='o', label=f'Condition {condition}', alpha=0.8)
                ax.fill_between(bin_centers[:len(lifespan_stats)], 
                               lifespan_stats['mean'] - lifespan_stats['sem'],
                               lifespan_stats['mean'] + lifespan_stats['sem'], 
                               alpha=0.2)
            
            ax.set_xlabel('Lifespan Percentage (%)')
            ax.set_ylabel(feature.replace('_', ' ').title())
            ax.set_title(f'{feature.replace("_", " ").title()} vs Lifespan %')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks([0, 25, 50, 75, 100])
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'temporal_trends_key_features.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Heatmap of all feature trends
    create_trend_heatmap(df, features, plot_dir)
    
    # 3. Individual feature plots for most significant trends
    create_significant_trend_plots(df, features, plot_dir)
    
    # 4. End-of-life focused analysis
    create_end_of_life_analysis(df, features, plot_dir)

def create_trend_heatmap(df, features, plot_dir):
    """Create a heatmap showing trend directions and significance."""
    trend_data = []
    
    for feature in features:
        for condition in df['label'].unique():
            condition_df = df[df['label'] == condition]
            lifespan_means = condition_df.groupby('lifespan_bin')[feature].mean()
            
            if len(lifespan_means) < 3:
                continue
                
            # Use bin centers for regression
            bin_centers = np.array([10, 30, 50, 70, 90])[:len(lifespan_means)]
            values = lifespan_means.values
            
            # Remove NaN values
            valid_mask = ~np.isnan(values)
            if valid_mask.sum() < 3:
                continue
                
            clean_centers = bin_centers[valid_mask]
            clean_values = values[valid_mask]
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(clean_centers, clean_values)
            
            # Create trend score (slope weighted by significance)
            trend_score = slope * (1 if p_value < 0.05 else 0.5)
            
            trend_data.append({
                'feature': feature,
                'condition': f'Condition {condition}',
                'trend_score': trend_score,
                'p_value': p_value,
                'r_squared': r_value**2
            })
    
    if trend_data:
        trend_df = pd.DataFrame(trend_data)
        pivot_df = trend_df.pivot(index='feature', columns='condition', values='trend_score')
        
        plt.figure(figsize=(12, len(features) * 0.4))
        sns.heatmap(pivot_df, annot=True, cmap='RdBu_r', center=0, 
                   fmt='.3f', cbar_kws={'label': 'Trend Score'})
        plt.title('Lifespan Trends Heatmap\n(Positive = Increasing with age, Negative = Decreasing with age)')
        plt.ylabel('Features')
        plt.xlabel('Conditions')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'temporal_trends_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

def create_significant_trend_plots(df, features, plot_dir):
    """Create individual plots for features with significant temporal trends."""
    significant_features = []
    
    # Find features with significant trends
    for feature in features:
        lifespan_means = df.groupby('lifespan_bin')[feature].mean()
        if len(lifespan_means) < 3:
            continue
            
        bin_centers = np.array([10, 30, 50, 70, 90])[:len(lifespan_means)]
        values = lifespan_means.values
        
        valid_mask = ~np.isnan(values)
        if valid_mask.sum() < 3:
            continue
            
        clean_centers = bin_centers[valid_mask]
        clean_values = values[valid_mask]
        
        _, _, r_value, p_value, _ = stats.linregress(clean_centers, clean_values)
        
        if p_value < 0.05 and r_value**2 > 0.1:  # Significant and meaningful
            significant_features.append((feature, p_value, r_value**2))
    
    # Sort by significance
    significant_features.sort(key=lambda x: x[1])
    
    # Plot top 6 most significant features
    if significant_features:
        n_plots = min(6, len(significant_features))
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (feature, p_val, r2) in enumerate(significant_features[:n_plots]):
            ax = axes[i]
            
            # Plot raw data points
            for condition in df['label'].unique():
                condition_df = df[df['label'] == condition]
                ax.scatter(condition_df['lifespan_percentage'], condition_df[feature], 
                          alpha=0.3, s=10, label=f'Condition {condition}')
            
            # Plot trend lines
            for condition in df['label'].unique():
                condition_df = df[df['label'] == condition]
                lifespan_means = condition_df.groupby('lifespan_bin')[feature].mean()
                
                if len(lifespan_means) >= 3:
                    bin_centers = np.array([10, 30, 50, 70, 90])[:len(lifespan_means)]
                    ax.plot(bin_centers, lifespan_means.values, 
                           linewidth=2, marker='o', label=f'Trend {condition}')
            
            ax.set_xlabel('Lifespan Percentage (%)')
            ax.set_ylabel(feature.replace('_', ' ').title())
            ax.set_title(f'{feature.replace("_", " ").title()}\n(p={p_val:.3f}, R²={r2:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks([0, 25, 50, 75, 100])
        
        # Hide empty subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'significant_temporal_trends.png'), dpi=300, bbox_inches='tight')
        plt.close()

def create_end_of_life_analysis(df, features, plot_dir):
    """Create focused analysis on end-of-life patterns (last 20% of lifespan)."""
    print("Creating end-of-life focused analysis...")
    
    # Focus on end-of-life segments (80-100% of lifespan)
    end_of_life_df = df[df['lifespan_percentage'] >= 80].copy()
    
    if len(end_of_life_df) == 0:
        print("No end-of-life segments found for analysis.")
        return
    
    # Create fine-grained bins for end-of-life analysis
    end_of_life_df['eol_bin'] = pd.cut(end_of_life_df['lifespan_percentage'], 
                                      bins=[80, 85, 90, 95, 100], 
                                      labels=['80-85%', '85-90%', '90-95%', '95-100%'])
    
    # Key behavioral features for end-of-life analysis
    eol_features = ['mean_speed', 'fraction_paused', 'mean_frenetic_score', 
                   'mean_roaming_score', 'activity_level', 'movement_efficiency']
    available_eol_features = [f for f in eol_features if f in features]
    
    if len(available_eol_features) >= 3:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(available_eol_features[:6]):
            ax = axes[i]
            
            # Plot for each condition
            for condition in end_of_life_df['label'].unique():
                condition_df = end_of_life_df[end_of_life_df['label'] == condition]
                eol_stats = condition_df.groupby('eol_bin')[feature].agg(['mean', 'sem']).reset_index()
                
                # Convert bin labels to numeric for plotting
                bin_centers = [82.5, 87.5, 92.5, 97.5]  # Center of each 5% bin
                ax.plot(bin_centers[:len(eol_stats)], eol_stats['mean'], 
                       marker='o', linewidth=2, label=f'Condition {condition}', alpha=0.8)
                ax.fill_between(bin_centers[:len(eol_stats)], 
                               eol_stats['mean'] - eol_stats['sem'],
                               eol_stats['mean'] + eol_stats['sem'], 
                               alpha=0.2)
            
            ax.set_xlabel('Lifespan Percentage (%)')
            ax.set_ylabel(feature.replace('_', ' ').title())
            ax.set_title(f'{feature.replace("_", " ").title()} - End of Life')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks([80, 85, 90, 95, 100])
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'end_of_life_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save end-of-life statistics
    eol_stats = {}
    for feature in available_eol_features:
        eol_means = end_of_life_df.groupby('eol_bin')[feature].mean()
        eol_stats[feature] = {
            '80-85%': eol_means.get('80-85%', np.nan),
            '85-90%': eol_means.get('85-90%', np.nan),
            '90-95%': eol_means.get('90-95%', np.nan),
            '95-100%': eol_means.get('95-100%', np.nan)
        }
    
    eol_df = pd.DataFrame(eol_stats).T
    eol_df.to_csv(os.path.join(plot_dir, 'end_of_life_statistics.csv'))
    
    print(f"End-of-life analysis complete. Analyzed {len(end_of_life_df)} segments from {len(end_of_life_df['original_file'].unique())} worms.")

def analyze_first_vs_last_50_percent(df, features, plot_dir):
    """Analyze features with the most dramatic differences between first 50% and last 50% of life."""
    print("\n=== First 50% vs Last 50% Life Analysis ===")
    
    # Split data into first 50% and last 50% of life
    first_50_df = df[df['lifespan_percentage'] <= 50]
    last_50_df = df[df['lifespan_percentage'] > 50]
    
    print(f"First 50% of life: {len(first_50_df)} segments")
    print(f"Last 50% of life: {len(last_50_df)} segments")
    
    # Analyze each feature
    comparison_results = {}
    
    for feature in features:
        if feature in first_50_df.columns and feature in last_50_df.columns:
            first_mean = first_50_df[feature].mean()
            last_mean = last_50_df[feature].mean()
            
            # Calculate percentage change
            if first_mean != 0:
                pct_change = ((last_mean - first_mean) / abs(first_mean)) * 100
            else:
                pct_change = np.nan
            
            # T-test for statistical significance
            try:
                t_stat, p_value = stats.ttest_ind(first_50_df[feature].dropna(), 
                                                last_50_df[feature].dropna())
                
                comparison_results[feature] = {
                    'first_50_mean': first_mean,
                    'last_50_mean': last_mean,
                    'absolute_difference': last_mean - first_mean,
                    'percentage_change': pct_change,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': abs(t_stat) / np.sqrt(len(first_50_df) + len(last_50_df) - 2)
                }
            except:
                continue
    
    # Sort by percentage change magnitude
    sorted_features = sorted(comparison_results.items(), 
                           key=lambda x: abs(x[1]['percentage_change']) if not np.isnan(x[1]['percentage_change']) else 0, 
                           reverse=True)
    
    # Save results
    comparison_df = pd.DataFrame(comparison_results).T
    comparison_df.to_csv(os.path.join(plot_dir, 'first_vs_last_50_percent_analysis.csv'))
    
    # Print top results
    print(f"\nTop 10 features with largest changes between first 50% and last 50% of life:")
    print("=" * 80)
    print(f"{'Feature':<25} {'First 50%':<12} {'Last 50%':<12} {'Change %':<10} {'p-value':<8} {'Significant'}")
    print("=" * 80)
    
    for feature, stats in sorted_features[:10]:
        first_val = f"{stats['first_50_mean']:.3f}"
        last_val = f"{stats['last_50_mean']:.3f}"
        pct = f"{stats['percentage_change']:.1f}%" if not np.isnan(stats['percentage_change']) else "N/A"
        p_val = f"{stats['p_value']:.3f}"
        sig = "***" if stats['significant'] else "   "
        
        print(f"{feature:<25} {first_val:<12} {last_val:<12} {pct:<10} {p_val:<8} {sig}")
    
    # Create visualization of top features
    create_first_vs_last_50_visualization(df, sorted_features[:10], plot_dir)
    
    return comparison_results

def create_first_vs_last_50_visualization(df, top_features, plot_dir):
    """Create visualization of the top features showing first vs last 50% differences."""
    if not top_features:
        return
    
    # Create a multi-panel plot
    n_features = min(6, len(top_features))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (feature, stats) in enumerate(top_features[:n_features]):
        ax = axes[i]
        
        # Create violin plots for first 50% vs last 50%
        first_50_data = df[df['lifespan_percentage'] <= 50][feature].dropna()
        last_50_data = df[df['lifespan_percentage'] > 50][feature].dropna()
        
        # Create violin plot
        violin_parts = ax.violinplot([first_50_data, last_50_data], 
                                   positions=[1, 2], 
                                   showmeans=True, 
                                   showmedians=True)
        
        # Color the violins
        violin_parts['bodies'][0].set_facecolor('lightblue')
        violin_parts['bodies'][1].set_facecolor('lightcoral')
        
        # Add box plots on top
        bp1 = ax.boxplot(first_50_data, positions=[1], widths=0.3, patch_artist=True)
        bp2 = ax.boxplot(last_50_data, positions=[2], widths=0.3, patch_artist=True)
        
        bp1['boxes'][0].set_facecolor('lightblue')
        bp2['boxes'][0].set_facecolor('lightcoral')
        
        # Customize plot
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['First 50%', 'Last 50%'])
        ax.set_ylabel(feature.replace('_', ' ').title())
        ax.set_title(f'{feature.replace("_", " ").title()}\n{stats["percentage_change"]:.1f}% change (p={stats["p_value"]:.3f})')
        ax.grid(True, alpha=0.3)
        
        # Add significance indicator
        if stats['significant']:
            ax.text(1.5, ax.get_ylim()[1] * 0.95, '***', ha='center', va='top', 
                   fontsize=16, fontweight='bold', color='red')
    
    # Hide empty subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'first_vs_last_50_percent_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved: first_vs_last_50_percent_comparison.png")

def perform_temporal_statistical_analysis(df, features):
    """Perform statistical analysis of temporal patterns."""
    results = {
        'significant_increasing_trends': [],
        'significant_decreasing_trends': [],
        'condition_differences': {},
        'early_vs_late_analysis': {},
        'end_of_life_analysis': {}
    }
    
    # Analyze trends for significance
    for feature in features:
        lifespan_means = df.groupby('lifespan_bin')[feature].mean()
        if len(lifespan_means) < 3:
            continue
            
        bin_centers = np.array([10, 30, 50, 70, 90])[:len(lifespan_means)]
        values = lifespan_means.values
        
        valid_mask = ~np.isnan(values)
        if valid_mask.sum() < 3:
            continue
        
        clean_centers = bin_centers[valid_mask]
        clean_values = values[valid_mask]
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(clean_centers, clean_values)
        
        if p_value < 0.05:
            trend_info = {
                'feature': feature,
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'percentage_change': ((clean_values[-1] - clean_values[0]) / abs(clean_values[0])) * 100 if clean_values[0] != 0 else np.nan
            }
            
            if slope > 0:
                results['significant_increasing_trends'].append(trend_info)
            else:
                results['significant_decreasing_trends'].append(trend_info)
    
    # Early vs late life analysis (first 20% vs last 20%)
    early_df = df[df['lifespan_percentage'] <= 20]
    late_df = df[df['lifespan_percentage'] >= 80]
    
    for feature in features:
        if feature in early_df.columns and feature in late_df.columns:
            early_mean = early_df[feature].mean()
            late_mean = late_df[feature].mean()
            
            # T-test for difference
            try:
                t_stat, p_value = stats.ttest_ind(early_df[feature].dropna(), 
                                                late_df[feature].dropna())
                
                results['early_vs_late_analysis'][feature] = {
                    'early_life_mean': early_mean,
                    'late_life_mean': late_mean,
                    'difference': late_mean - early_mean,
                    'percentage_change': ((late_mean - early_mean) / abs(early_mean)) * 100 if early_mean != 0 else np.nan,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            except:
                continue
    
    # End-of-life analysis (last 20% of lifespan)
    end_of_life_df = df[df['lifespan_percentage'] >= 80]
    if len(end_of_life_df) > 0:
        # Create fine-grained bins for end-of-life
        end_of_life_df['eol_bin'] = pd.cut(end_of_life_df['lifespan_percentage'], 
                                          bins=[80, 85, 90, 95, 100], 
                                          labels=['80-85%', '85-90%', '90-95%', '95-100%'])
        
        for feature in features:
            if feature in end_of_life_df.columns:
                eol_means = end_of_life_df.groupby('eol_bin')[feature].mean()
                
                if len(eol_means) >= 2:
                    # Calculate trend within end-of-life period
                    bin_centers = np.array([82.5, 87.5, 92.5, 97.5])[:len(eol_means)]
                    values = eol_means.values
                    
                    valid_mask = ~np.isnan(values)
                    if valid_mask.sum() >= 2:
                        clean_centers = bin_centers[valid_mask]
                        clean_values = values[valid_mask]
                        
                        try:
                            slope, intercept, r_value, p_value, std_err = stats.linregress(clean_centers, clean_values)
                            
                            results['end_of_life_analysis'][feature] = {
                                'slope': slope,
                                'r_squared': r_value**2,
                                'p_value': p_value,
                                'significant': p_value < 0.05,
                                'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                                'final_20_percent_change': ((clean_values[-1] - clean_values[0]) / abs(clean_values[0])) * 100 if clean_values[0] != 0 else np.nan
                            }
                        except:
                            continue
    
    return results

def save_temporal_analysis_results(results, plot_dir):
    """Save analysis results to files."""
    
    # Save overall trends
    if 'overall_trends' in results:
        overall_df = pd.DataFrame(results['overall_trends']).T
        overall_df.to_csv(os.path.join(plot_dir, 'overall_temporal_trends.csv'))
    
    # Save condition-specific trends
    if 'condition_trends' in results:
        for condition, trends in results['condition_trends'].items():
            condition_df = pd.DataFrame(trends).T
            condition_df.to_csv(os.path.join(plot_dir, f'{condition}_temporal_trends.csv'))
    
    # Save statistical analysis
    if 'statistical_analysis' in results:
        stat_results = results['statistical_analysis']
        
        # Save early vs late analysis
        if 'early_vs_late_analysis' in stat_results:
            early_late_df = pd.DataFrame(stat_results['early_vs_late_analysis']).T
            early_late_df.to_csv(os.path.join(plot_dir, 'early_vs_late_life_analysis.csv'))
        
        # Save end-of-life analysis
        if 'end_of_life_analysis' in stat_results:
            eol_df = pd.DataFrame(stat_results['end_of_life_analysis']).T
            eol_df.to_csv(os.path.join(plot_dir, 'end_of_life_trends.csv'))
        
        # Save significant trends
        if stat_results['significant_increasing_trends']:
            inc_df = pd.DataFrame(stat_results['significant_increasing_trends'])
            inc_df.to_csv(os.path.join(plot_dir, 'significant_increasing_trends.csv'), index=False)
        
        if stat_results['significant_decreasing_trends']:
            dec_df = pd.DataFrame(stat_results['significant_decreasing_trends'])
            dec_df.to_csv(os.path.join(plot_dir, 'significant_decreasing_trends.csv'), index=False)

        # Save first vs last 50% analysis
        if 'first_vs_last_50_percent_analysis' in stat_results:
            first_vs_last_df = pd.DataFrame(stat_results['first_vs_last_50_percent_analysis']).T
            first_vs_last_df.to_csv(os.path.join(plot_dir, 'first_vs_last_50_percent_analysis.csv'))

def print_summary_findings(results):
    """Print a summary of key findings."""
    print("\n" + "="*80)
    print("TEMPORAL ANALYSIS SUMMARY - RELATIVE LIFESPAN PERCENTAGE")
    print("="*80)
    
    if 'statistical_analysis' in results:
        stat_results = results['statistical_analysis']
        
        print(f"\nFeatures with significant INCREASING trends with age: {len(stat_results['significant_increasing_trends'])}")
        for trend in stat_results['significant_increasing_trends'][:5]:  # Top 5
            print(f"  - {trend['feature']}: {trend['percentage_change']:.1f}% change (p={trend['p_value']:.3f})")
        
        print(f"\nFeatures with significant DECREASING trends with age: {len(stat_results['significant_decreasing_trends'])}")
        for trend in stat_results['significant_decreasing_trends'][:5]:  # Top 5
            print(f"  - {trend['feature']}: {trend['percentage_change']:.1f}% change (p={trend['p_value']:.3f})")
        
        print(f"\nEarly vs Late Life Analysis (first 20% vs last 20%):")
        if 'early_vs_late_analysis' in stat_results:
            significant_changes = [(k, v) for k, v in stat_results['early_vs_late_analysis'].items() 
                                 if v['significant']]
            print(f"  Features with significant changes: {len(significant_changes)}")
            
            for feature, data in significant_changes[:5]:
                print(f"  - {feature}: {data['percentage_change']:.1f}% change (p={data['p_value']:.3f})")
        
        print(f"\nEnd-of-Life Analysis (last 20% of lifespan):")
        if 'end_of_life_analysis' in stat_results:
            significant_eol = [(k, v) for k, v in stat_results['end_of_life_analysis'].items() 
                             if v['significant']]
            print(f"  Features with significant trends in final 20%: {len(significant_eol)}")
            
            for feature, data in significant_eol[:5]:
                direction = "increasing" if data['trend_direction'] == 'increasing' else "decreasing"
                print(f"  - {feature}: {direction} trend, {data['final_20_percent_change']:.1f}% change (p={data['p_value']:.3f})")

        print(f"\nFirst 50% vs Last 50% Life Analysis:")
        if 'first_vs_last_50_percent_analysis' in stat_results:
            first_vs_last_df = pd.DataFrame(stat_results['first_vs_last_50_percent_analysis']).T
            if not first_vs_last_df.empty:
                top_features = sorted(stat_results['first_vs_last_50_percent_analysis'].items(), 
                                     key=lambda x: abs(x[1]['percentage_change']) if not np.isnan(x[1]['percentage_change']) else 0, 
                                     reverse=True)[:10]
                print(f"  Top 10 features with largest changes between first 50% and last 50% of life:")
                print("=" * 80)
                print(f"{'Feature':<25} {'First 50%':<12} {'Last 50%':<12} {'Change %':<10} {'p-value':<8} {'Significant'}")
                print("=" * 80)
                for feature, stats in top_features:
                    first_val = f"{stats['first_50_mean']:.3f}"
                    last_val = f"{stats['last_50_mean']:.3f}"
                    pct = f"{stats['percentage_change']:.1f}%" if not np.isnan(stats['percentage_change']) else "N/A"
                    p_val = f"{stats['p_value']:.3f}"
                    sig = "***" if stats['significant'] else "   "
                    print(f"{feature:<25} {first_val:<12} {last_val:<12} {pct:<10} {p_val:<8} {sig}")

def main():
    """Main function to run temporal analysis."""
    parser = argparse.ArgumentParser(description='Analyze temporal patterns in segment features')
    parser.add_argument('--data_type', default='segments', choices=['full', 'segments'],
                       help='Type of data to analyze')
    
    args = parser.parse_args()
    
    # Set up plotting
    setup_plotting_style()
    
    # Load data
    df, plot_dir = load_data(args.data_type)
    
    # Run analysis
    results = {}
    analyze_temporal_trends(df, plot_dir)
    
    # Print summary findings
    print_summary_findings(results)
    
    print(f"\nTemporal analysis complete! Results saved to {plot_dir}")
    print("\nKey files generated:")
    print("- temporal_trends_key_features.png: Overview of key behavioral trends across lifespan")
    print("- temporal_trends_heatmap.png: Heatmap of all feature trends with age")
    print("- significant_temporal_trends.png: Detailed plots of significant trends")
    print("- end_of_life_analysis.png: Focused analysis of final 20% of lifespan")
    print("- overall_temporal_trends.csv: Statistical summary of trends")
    print("- early_vs_late_life_analysis.csv: Comparison of early vs late life")
    print("- end_of_life_trends.csv: Trends specifically in final 20% of lifespan")
    print("- end_of_life_statistics.csv: Detailed statistics for end-of-life period")
    print("- first_vs_last_50_percent_analysis.csv: Features with largest changes between first 50% and last 50% of life")
    print("- first_vs_last_50_percent_comparison.png: Visualization of top features")

if __name__ == "__main__":
    main() 