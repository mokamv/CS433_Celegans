"""
Polynomial regression to predict segments remaining until death using mean_speed.
Data source: feature_data/segments_features.csv.
Target extraction mirrors logic from death_proximity_regressor.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(features_csv_path, selected_features=None):
	"""
	Load features and compute target as 'segments_from_end' using segment indices.

	- If 'segment_index' is missing, extract it from 'filename' using the pattern
	  ...-segment{index}-preprocessed.csv as in death_proximity_regressor.
	- Compute max segment per worm using 'original_file'.
	- Target y = max_segment_index - segment_index (segments remaining until death).
	- If selected_features is None or 'all', uses all numeric features except metadata columns.

	Returns X (selected features), y (segments_from_end), and the cleaned DataFrame.
	"""
	features_csv_path = Path(features_csv_path)
	df = pd.read_csv(features_csv_path)

	# Exclude metadata columns
	metadata_cols = {'filename', 'segment_index', 'original_file', 'max_segment_index', 'segments_from_end'}
	
	# If no features specified or 'all' requested, use all numeric columns except metadata
	if not selected_features or selected_features == 'all':
		numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
		selected_features = [col for col in numeric_cols if col not in metadata_cols]
		print(f"Using all available features: {selected_features}")
	
	# Validate requested features exist
	missing_feats = [f for f in selected_features if f not in df.columns]
	if missing_feats:
		raise ValueError(f"Missing requested feature columns in features CSV: {missing_feats}")

	# Ensure segment_index exists; extract from filename if needed
	if 'segment_index' not in df.columns or df['segment_index'].isna().all():
		df['segment_index'] = df['filename'].str.extract(r'segment(\d+(?:\.\d+)?)', expand=False).astype(float)

	# Validate necessary metadata
	if 'original_file' not in df.columns:
		raise ValueError("Missing required column 'original_file' to group segments per worm")

	# Compute max segment index per worm and segments_from_end
	max_segments = df.groupby('original_file')['segment_index'].max()
	df['max_segment_index'] = df['original_file'].map(max_segments)
	df['segments_from_end'] = df['max_segment_index'] - df['segment_index']

	# Drop rows with missing values in required columns
	required = list(selected_features) + ['segment_index', 'segments_from_end']
	df_clean = df.dropna(subset=required).copy()

	print(f"Loaded features: {len(df)} rows, using {len(df_clean)} after cleaning")
	print(f"Target: segments_from_end | Range: [{df_clean['segments_from_end'].min():.2f}, {df_clean['segments_from_end'].max():.2f}] | Mean: {df_clean['segments_from_end'].mean():.2f}")

	X = df_clean[selected_features].to_numpy()
	y = df_clean['segments_from_end'].to_numpy()

	return X, y, df_clean, selected_features


def plot_mean_speed_vs_segments(df: pd.DataFrame, save_path: Path = None, show: bool = True):
	"""Plot mean_speed against segments_from_end.

	Requires columns 'mean_speed' and 'segments_from_end' in df.
	"""
	import matplotlib.pyplot as plt

	if 'mean_speed' not in df.columns or 'segments_from_end' not in df.columns:
		print("Plot skipped: 'mean_speed' or 'segments_from_end' missing in DataFrame")
		return

	ms = df['mean_speed'].to_numpy()
	seg = df['segments_from_end'].to_numpy()

	plt.figure(figsize=(8, 5))
	plt.scatter(ms, seg, s=12, alpha=0.5)
	plt.xlabel('Mean Speed')
	plt.ylabel('Segments from End')
	plt.title('Mean Speed vs Segments Remaining Until Death')
	plt.grid(True, alpha=0.3)

	if save_path is not None:
		try:
			plt.tight_layout()
			plt.savefig(save_path, dpi=200)
			print(f"Saved plot to: {save_path}")
		except Exception as e:
			print(f"Could not save plot: {e}")
	if show:
		plt.show()
	else:
		plt.close()


def plot_averaged_speed_vs_segments(df: pd.DataFrame, window_size: int = 5, save_path: Path = None, show: bool = True):
	"""Plot averaged mean_speed over N consecutive segments vs segments_from_end.

	Groups by worm (original_file), sorts by segment_index, then computes rolling average.
	
	Args:
		df: DataFrame with columns 'original_file', 'segment_index', 'mean_speed', 'segments_from_end'
		window_size: Number of consecutive segments to average over
		save_path: Optional path to save the plot
	"""
	import matplotlib.pyplot as plt

	required_cols = ['original_file', 'segment_index', 'mean_speed', 'segments_from_end']
	if not all(col in df.columns for col in required_cols):
		print(f"Plot skipped: Missing required columns {required_cols}")
		return

	# Make a copy and sort by worm and segment index
	df_sorted = df[required_cols].copy().sort_values(['original_file', 'segment_index'])

	# Compute rolling average within each worm
	df_sorted['mean_speed_avg'] = df_sorted.groupby('original_file')['mean_speed'].transform(
		lambda x: x.rolling(window=window_size, min_periods=1).mean()
	)

	# Plot
	plt.figure(figsize=(10, 6))
	plt.scatter(df_sorted['mean_speed_avg'], df_sorted['segments_from_end'], s=20, alpha=0.6)
	plt.xlabel(f'Mean Speed (averaged over {window_size} segments)')
	plt.ylabel('Segments from End')
	plt.title(f'Averaged Mean Speed vs Segments Remaining Until Death (window={window_size})')
	plt.grid(True, alpha=0.3)

	if save_path is not None:
		try:
			plt.tight_layout()
			plt.savefig(save_path, dpi=200)
			print(f"Saved plot to: {save_path}")
		except Exception as e:
			print(f"Could not save plot: {e}")
	if show:
		plt.show()
	else:
		plt.close()




def train_polynomial_regression(X, y, feature_names, degree=2, alpha=1.0, test_size=0.2, random_state=42, n_splits=5, groups=None, regularization='l2'):
	"""
	Train a polynomial regression model using GroupKFold cross-validation.
	
	Args:
		X: Feature matrix
		y: Target vector
		feature_names: Names of features
		degree: Polynomial degree
		alpha: Regularization strength
		test_size: Test split size (unused, kept for compatibility)
		random_state: Random seed
		n_splits: Number of folds for GroupKFold
		groups: Group labels for GroupKFold (e.g., worm IDs) - required
		regularization: Type of regularization ('l2' for Ridge, 'l1' for Lasso)
	"""
	if groups is None:
		raise ValueError("GroupKFold requires groups parameter (worm IDs). Cannot train without it.")
	
	if regularization.lower() not in ['l1', 'l2']:
		raise ValueError(f"regularization must be 'l1' or 'l2', got {regularization}")
	
	poly = PolynomialFeatures(degree=degree, include_bias=False)
	scaler = StandardScaler()
	
	# Choose regularization model
	if regularization.lower() == 'l2':
		regression_model = Ridge(alpha=alpha, max_iter=5000)
		reg_name = 'Ridge (L2)'
	else:
		regression_model = Lasso(alpha=alpha, max_iter=5000, random_state=random_state)
		reg_name = 'Lasso (L1)'
	
	# Group K-Fold Cross-Validation
	gkf = GroupKFold(n_splits=n_splits)

	fold_metrics = {
		'train_rmse': [], 'test_rmse': [],
		'train_mae': [], 'test_mae': [],
		'train_r2': [], 'test_r2': []
	}

	# Collect per-sample true/pred pairs (test folds) for downstream analysis/plots
	y_true_pred = []
	
	print(f"\nGroup K-Fold Cross-Validation (k={n_splits}):")
	print(f"  Degree: {degree} | {reg_name} alpha: {alpha}")
	print(f"  Feature scaling: StandardScaler(mean/std)")
	
	# Generate splits
	splits = gkf.split(X, y, groups)
	
	for fold_num, (train_idx, test_idx) in enumerate(splits):
		X_train_fold, X_test_fold = X[train_idx], X[test_idx]
		y_train_fold, y_test_fold = y[train_idx], y[test_idx]
		
		# Transform features
		X_train_poly = poly.fit_transform(X_train_fold)
		X_test_poly = poly.transform(X_test_fold)
		
		# Standardize
		X_train_poly_std = scaler.fit_transform(X_train_poly)
		X_test_poly_std = scaler.transform(X_test_poly)
		
		# Train
		regression_model.fit(X_train_poly_std, y_train_fold)
		
		# Predict
		y_train_pred = regression_model.predict(X_train_poly_std)
		y_test_pred = regression_model.predict(X_test_poly_std)

		# Save per-sample true/pred for test fold
		y_true_pred.append((y_test_fold.copy(), y_test_pred.copy()))
		
		# Metrics
		fold_metrics['train_rmse'].append(np.sqrt(mean_squared_error(y_train_fold, y_train_pred)))
		fold_metrics['test_rmse'].append(np.sqrt(mean_squared_error(y_test_fold, y_test_pred)))
		fold_metrics['train_mae'].append(mean_absolute_error(y_train_fold, y_train_pred))
		fold_metrics['test_mae'].append(mean_absolute_error(y_test_fold, y_test_pred))
		fold_metrics['train_r2'].append(r2_score(y_train_fold, y_train_pred))
		fold_metrics['test_r2'].append(r2_score(y_test_fold, y_test_pred))
		
		print(f"  Fold {fold_num + 1}: Test RMSE={fold_metrics['test_rmse'][-1]:.4f}, Test MAE={fold_metrics['test_mae'][-1]:.4f}, Test R²={fold_metrics['test_r2'][-1]:.4f}")
	
	# Retrain on full data for final model
	X_poly = poly.fit_transform(X)
	X_poly_std = scaler.fit_transform(X_poly)
	regression_model.fit(X_poly_std, y)
	
	try:
		poly_feature_names = poly.get_feature_names_out(feature_names).tolist()
	except Exception:
		poly_feature_names = [f"f{i}" for i in range(X_poly.shape[1])]
	
	print(f"\nGroup K-Fold Results Summary:")
	print(f"  Train RMSE: {np.mean(fold_metrics['train_rmse']):.4f} ± {np.std(fold_metrics['train_rmse']):.4f}")
	print(f"  Test RMSE: {np.mean(fold_metrics['test_rmse']):.4f} ± {np.std(fold_metrics['test_rmse']):.4f}")
	print(f"  Train MAE: {np.mean(fold_metrics['train_mae']):.4f} ± {np.std(fold_metrics['train_mae']):.4f}")
	print(f"  Test MAE: {np.mean(fold_metrics['test_mae']):.4f} ± {np.std(fold_metrics['test_mae']):.4f}")
	print(f"  Train R²: {np.mean(fold_metrics['train_r2']):.4f} ± {np.std(fold_metrics['train_r2']):.4f}")
	print(f"  Test R²: {np.mean(fold_metrics['test_r2']):.4f} ± {np.std(fold_metrics['test_r2']):.4f}")
	
	return {
		'model': regression_model,
		'poly': poly,
		'scaler': scaler,
		'coefficients': regression_model.coef_,
		'intercept': regression_model.intercept_,
		'n_splits': n_splits,
		'fold_metrics': fold_metrics,
		'train_rmse': np.mean(fold_metrics['train_rmse']),
		'test_rmse': np.mean(fold_metrics['test_rmse']),
		'train_mae': np.mean(fold_metrics['train_mae']),
		'test_mae': np.mean(fold_metrics['test_mae']),
		'train_r2': np.mean(fold_metrics['train_r2']),
		'test_r2': np.mean(fold_metrics['test_r2']),
		'y_true_pred': y_true_pred,
	}


def plot_error_by_true_bins(y_true_pred, bin_width=20, n_bins=None, save_path=None, show: bool = True):
	"""Plot RMSE and MAE grouped by true segments-from-death bins and print stats."""
	import matplotlib.pyplot as plt

	if not y_true_pred:
		print("No predictions available to plot.")
		return None

	# Flatten arrays
	y_true_all = np.concatenate([yt for yt, _ in y_true_pred])
	y_pred_all = np.concatenate([yp for _, yp in y_true_pred])

	max_true = y_true_all.max()

	if n_bins is not None and n_bins > 0:
		bin_edges = np.linspace(0, max_true, n_bins + 1)
		bin_label = f"{n_bins} bins"
	else:
		bin_edges = np.arange(0, max_true + bin_width, bin_width)
		bin_label = f"bin width = {bin_width}"

	bin_centers = []
	rmse_vals = []
	mae_vals = []
	counts = []

	print("\nError-by-bin breakdown (true segments-from-death):")
	print(f"{'Bin range':<18} {'Count':<8} {'RMSE':<10} {'MAE':<10}")
	for b_start, b_end in zip(bin_edges[:-1], bin_edges[1:]):
		mask = (y_true_all >= b_start) & (y_true_all < b_end)
		if not np.any(mask):
			continue
		y_true_bin = y_true_all[mask]
		y_pred_bin = y_pred_all[mask]
		err = y_pred_bin - y_true_bin
		rmse_val = np.sqrt(np.mean(err ** 2))
		mae_val = np.mean(np.abs(err))
		rmse_vals.append(rmse_val)
		mae_vals.append(mae_val)
		bin_centers.append((b_start + b_end) / 2)
		counts.append(len(y_true_bin))
		print(f"[{b_start:>6.1f}, {b_end:>6.1f})   {len(y_true_bin):<8d} {rmse_val:<10.4f} {mae_val:<10.4f}")

	plt.figure(figsize=(8, 5))
	plt.plot(bin_centers, rmse_vals, marker='o', label='RMSE')
	plt.plot(bin_centers, mae_vals, marker='s', label='MAE')
	plt.xlabel(f'True segments from death ({bin_label})')
	plt.ylabel('Error')
	plt.title('Error by True Segments-from-Death Bin')
	plt.grid(True, alpha=0.3)
	plt.legend()

	if save_path is not None:
		try:
			plt.tight_layout()
			plt.savefig(save_path, dpi=200)
			print(f"Saved error-by-bin plot to: {save_path}")
		except Exception as e:
			print(f"Could not save error plot: {e}")

	plt.show()

	return {
		'bin_centers': np.array(bin_centers),
		'rmse': np.array(rmse_vals),
		'mae': np.array(mae_vals),
		'counts': np.array(counts),
		'bin_width': bin_width,
		'n_bins': n_bins,
	}


def main(metric: str = 'rmse', selected_features=None, alpha: float = 1.0, alpha_grid=None, n_splits=5, regularization='l2', polynomial_degrees=None, show_plots: bool = False, error_bin_count: int | None = None, error_bin_width: int = 20):
	"""Main execution function.

	Args:
		metric: Evaluation metric ('mae' or 'rmse')
		selected_features: List of feature names to use
		alpha: Regularization strength for polynomial regression
		alpha_grid: List of alpha values to grid search
		n_splits: Number of folds for GroupKFold CV
		regularization: Type of regularization ('l2' for Ridge, 'l1' for Lasso)
		polynomial_degrees: List of polynomial degrees to test (e.g., [1, 2, 3])
		show_plots: Whether to display exploratory plots (mean speed vs segments)
		error_bin_count: Override number of bins for the final error plot (if None, uses bin_width)
		error_bin_width: Bin width to use when error_bin_count is None
	"""
	base_dir = Path(__file__).parent
	features_csv = base_dir.parent / "feature_data" / "segments_features.csv"

	# Default polynomial degrees if not specified
	if polynomial_degrees is None:
		polynomial_degrees = [1, 2, 3]

	print("=" * 70)
	print("Polynomial Regression for Segments Remaining Until Death (mean_speed)")
	print("=" * 70)

	# Load data and compute target
	print("\nLoading features and computing segments_from_end...")
	X, y, metadata, actual_features = load_and_prepare_data(features_csv, selected_features=selected_features)
	# Quick visualization of mean_speed vs segments_from_end
	plot_mean_speed_vs_segments(metadata, save_path=base_dir / 'mean_speed_vs_segments.pdf', show=show_plots)
	# Plot averaged mean_speed over segments
	plot_averaged_speed_vs_segments(metadata, window_size=5, save_path=base_dir / 'averaged_speed_vs_segments.pdf', show=show_plots)
	# Use the actual feature names returned from load_and_prepare_data
	feature_names_used = actual_features
	
	# Extract groups (worm IDs) for GroupKFold
	if 'original_file' in metadata.columns:
		# Map worm names to numeric group indices
		worm_to_group = {worm: idx for idx, worm in enumerate(metadata['original_file'].unique())}
		groups = metadata['original_file'].map(worm_to_group).to_numpy()
		print(f"\nUsing GroupKFold with {len(worm_to_group)} groups (worms)")
	else:
		raise ValueError("Missing required column 'original_file' for GroupKFold")

	# Basic statistics
	print(f"\nData Statistics:")
	print(f"  Features selected: {selected_features}")
	print(f"  X shape: {X.shape}, y shape: {y.shape}")
	print(f"  Segments from end - Min: {y.min():.2f}, Max: {y.max():.2f}, Mean: {y.mean():.2f}")

	# Train models with different polynomial degrees and (optionally) an alpha grid
	if alpha_grid is None or len(alpha_grid) == 0:
		alpha_grid = [alpha]

	grid_results = {}
	best_by_degree = {}

	print("\n" + "=" * 70)
	print("Polynomial Regression")
	print("=" * 70)

	for degree in polynomial_degrees:
		print(f"\n" + "=" * 70)
		print(f"Training degree {degree} with {len(alpha_grid)} alpha value(s)")
		print("=" * 70)
		for a in alpha_grid:
			res = train_polynomial_regression(X, y, feature_names=feature_names_used, degree=degree, alpha=a, 
											   n_splits=n_splits, groups=groups, regularization=regularization)
			grid_results[(degree, a)] = res

		# pick best alpha for this degree
		if metric.lower() == 'mae':
			best_alpha = min(alpha_grid, key=lambda a: grid_results[(degree, a)]['test_mae'])
		else:
			best_alpha = min(alpha_grid, key=lambda a: grid_results[(degree, a)]['test_rmse'])
		best_by_degree[degree] = {'alpha': best_alpha, **grid_results[(degree, best_alpha)]}

	# Compare models
	print(f"\n" + "=" * 70)
	print("Step 3: Model Comparison")
	print("=" * 70)
	print(f"Selected metric: {metric.upper()}")
	print(f"Validation method: Group K-Fold CV (k={n_splits})")
	print(f"Regularization: {regularization.upper()}")
	print(f"{'Degree':<10} {'Alpha':<10} {'Train RMSE':<15} {'Test RMSE':<15} {'Train MAE':<15} {'Test MAE':<15} {'Train R²':<15} {'Test R²':<15}")
	print("-" * 70)
	for (deg, a), result in sorted(grid_results.items(), key=lambda kv: (kv[0][0], kv[0][1])):
		print(f"{deg:<10} {a:<10.4g} {result['train_rmse']:<15.4f} {result['test_rmse']:<15.4f} "
			  f"{result['train_mae']:<15.4f} {result['test_mae']:<15.4f} "
			  f"{result['train_r2']:<15.4f} {result['test_r2']:<15.4f}")

	print("\nBest alpha per degree (by selected metric):")
	print(f"{'Degree':<10} {'Best Alpha':<12} {'Test RMSE':<12} {'Test MAE':<12} {'Test R²':<10}")
	for deg in sorted(best_by_degree.keys()):
		br = best_by_degree[deg]
		print(f"{deg:<10} {br['alpha']:<12.4g} {br['test_rmse']:<12.4f} {br['test_mae']:<12.4f} {br['test_r2']:<10.4f}")

	# Identify best degree based on chosen metric
	if metric.lower() == 'mae':
		best_pair = min(grid_results.keys(), key=lambda k: grid_results[k]['test_mae'])
		best_value = grid_results[best_pair]['test_mae']
	else:
		best_pair = min(grid_results.keys(), key=lambda k: grid_results[k]['test_rmse'])
		best_value = grid_results[best_pair]['test_rmse']

	print(f"\nBest (degree, alpha) by {metric.upper()}: {best_pair} (value: {best_value:.4f})")

	# Plot error by true-bin for the best model and print stats
	best_result = grid_results[best_pair]
	plot_info = plot_error_by_true_bins(
		best_result['y_true_pred'],
		bin_width=error_bin_width,
		n_bins=error_bin_count,
		save_path=base_dir / 'error_by_true_bin.pdf'
	)

	# Compute RMSE/MAE restricted to 20-80 true segments-from-death
	y_true_all = np.concatenate([yt for yt, _ in best_result['y_true_pred']])
	y_pred_all = np.concatenate([yp for _, yp in best_result['y_true_pred']])
	range_mask = (y_true_all >= 20) & (y_true_all <= 80)
	if not np.any(range_mask):
		print("No samples with true segments-from-death in [20, 80] to report range metrics.")
	else:
		err_range = y_pred_all[range_mask] - y_true_all[range_mask]
		rmse_range = np.sqrt(np.mean(err_range ** 2))
		mae_range = np.mean(np.abs(err_range))
		print(f"\nRange metrics for true segments-from-death in [20, 80]:")
		print(f"  Count: {range_mask.sum()} | RMSE: {rmse_range:.4f} | MAE: {mae_range:.4f}")

	print("\n" + "=" * 70)
	print("Execution completed successfully!")
	print("=" * 70)

	return {
		'grid_results': grid_results,
		'best_by_degree': best_by_degree,
		'best_overall': {'pair': best_pair, 'value': best_value, 'metric': metric.lower()}
	}


if __name__ == "__main__":
	# Hardcoded configuration parameters
	SELECTED_METRIC = 'mae'  # choose 'mae' or 'rmse'
	# Choose features to use:
	# - None or 'all': Use all available numeric features from CSV
	# - List of feature names: e.g., ['mean_speed', 'mean_jerk', 'turning_entropy']
	SELECTED_FEATURES = ["mean_speed", "mean_turning_angle"]#'all'  # Change to None to use all features, or specify a list
	# Regularization strength (alpha > 0) for polynomial regression
	ALPHA = 1.0
	# Optionally sweep across alphas (log-spaced) for polynomial regression
	ALPHA_GRID = list(np.logspace(-3, 3, 20))
	# Number of folds for GroupKFold CV
	N_SPLITS = 5
	# Type of regularization: 'l2' for Ridge, 'l1' for Lasso
	REGULARIZATION = 'l2'
	# Polynomial degrees to test
	POLYNOMIAL_DEGREES = [1,2,3,4]
	# Whether to show exploratory mean speed plots when launching the script
	SHOW_PLOTS = False
	# Control error-plot binning: set ERROR_BIN_COUNT to an int to fix bin count; None reverts to fixed width
	ERROR_BIN_COUNT = None
	ERROR_BIN_WIDTH = 5

	results = main(metric=SELECTED_METRIC, selected_features=SELECTED_FEATURES, alpha=ALPHA, 
			   alpha_grid=ALPHA_GRID, n_splits=N_SPLITS, regularization=REGULARIZATION,
		   polynomial_degrees=POLYNOMIAL_DEGREES, show_plots=SHOW_PLOTS,
		   error_bin_count=ERROR_BIN_COUNT, error_bin_width=ERROR_BIN_WIDTH)
