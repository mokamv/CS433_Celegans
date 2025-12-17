'''
Time from death prediction

DECLARATION: ChatGPT was used to write code for generating plots. 
'''
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def load_and_prepare_data(features_csv_path, selected_features=None):
	features_csv_path = Path(features_csv_path)
	df = pd.read_csv(features_csv_path)

	metadata_cols = {'filename', 'segment_index', 'original_file', 'max_segment_index', 'segments_from_end'}
	
	if not selected_features or selected_features == 'all':
		numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
		selected_features = [col for col in numeric_cols if col not in metadata_cols]
		print(f"Using all available features: {selected_features}")
	
  # check features exist
	missing_feats = [f for f in selected_features if f not in df.columns]
	if missing_feats:
		raise ValueError(f"Missing requested feature columns in features CSV: {missing_feats}")

	if 'segment_index' not in df.columns or df['segment_index'].isna().all():
		df['segment_index'] = df['filename'].str.extract(r'segment(\d+(?:\.\d+)?)', expand=False).astype(float)

	if 'original_file' not in df.columns:
		raise ValueError("Missing required column 'original_file' to group segments per worm")

	max_segments = df.groupby('original_file')['segment_index'].max()
	df['max_segment_index'] = df['original_file'].map(max_segments)
	df['segments_from_end'] = df['max_segment_index'] - df['segment_index']

	required = list(selected_features) + ['segment_index', 'segments_from_end']
	df_clean = df.dropna(subset=required).copy()

	print(f"Loaded features: {len(df)} rows, using {len(df_clean)} after cleaning")
	print(f"Target: segments_from_end | Range: [{df_clean['segments_from_end'].min():.2f}, {df_clean['segments_from_end'].max():.2f}] | Mean: {df_clean['segments_from_end'].mean():.2f}")

	X = df_clean[selected_features].to_numpy()
	y = df_clean['segments_from_end'].to_numpy()

	return X, y, df_clean, selected_features


def plot_mean_speed_vs_segments(df: pd.DataFrame, save_path: Path = None, show: bool = True):
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
	required_cols = ['original_file', 'segment_index', 'mean_speed', 'segments_from_end']
	if not all(col in df.columns for col in required_cols):
		print(f"Plot skipped: Missing required columns {required_cols}")
		return

	# sort by worm and segment index
	df_sorted = df[required_cols].copy().sort_values(['original_file', 'segment_index'])

	# rolling average
	df_sorted['mean_speed_avg'] = df_sorted.groupby('original_file')['mean_speed'].transform(
		lambda x: x.rolling(window=window_size, min_periods=1).mean()
	)

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

def train_polynomial_regression(X, y, feature_names, degree=2, alpha=1.0, 
																random_state=42, n_splits=5, 
																groups=None, regularization='l2'):
	if groups is None:
		raise ValueError("GroupKFold requires groups parameter (worm IDs). Cannot train without it.")
	
	if regularization.lower() not in ['l1', 'l2']:
		raise ValueError(f"regularization must be 'l1' or 'l2', got {regularization}")
	
	poly = PolynomialFeatures(degree=degree, include_bias=False)
	scaler = StandardScaler()
	
	if regularization.lower() == 'l2':
		regression_model = Ridge(alpha=alpha, max_iter=5000)
		reg_name = 'Ridge (L2)'
	else:
		regression_model = Lasso(alpha=alpha, max_iter=5000, random_state=random_state)
		reg_name = 'Lasso (L1)'
	
	gkf = GroupKFold(n_splits=n_splits)

	fold_metrics = {
		'train_rmse': [], 'test_rmse': [],
		'train_mae': [], 'test_mae': [],
		'train_r2': [], 'test_r2': []
	}

	y_true_pred = []
	
	print(f"\nGroup K-Fold Cross-Validation (k={n_splits}):")
	print(f"  Degree: {degree} | {reg_name} alpha: {alpha}")
	print(f"  Feature scaling: StandardScaler(mean/std)")
	
	splits = gkf.split(X, y, groups)
	
	for fold_num, (train_idx, test_idx) in enumerate(splits):
		X_train_fold, X_test_fold = X[train_idx], X[test_idx]
		y_train_fold, y_test_fold = y[train_idx], y[test_idx]
		
		X_train_poly = poly.fit_transform(X_train_fold)
		X_test_poly = poly.transform(X_test_fold)
		
		X_train_poly_std = scaler.fit_transform(X_train_poly)
		X_test_poly_std = scaler.transform(X_test_poly)
		
		regression_model.fit(X_train_poly_std, y_train_fold)
		
		y_train_pred = regression_model.predict(X_train_poly_std)
		y_test_pred = regression_model.predict(X_test_poly_std)

		y_true_pred.append((y_test_fold.copy(), y_test_pred.copy()))
		
		fold_metrics['train_rmse'].append(np.sqrt(mean_squared_error(y_train_fold, y_train_pred)))
		fold_metrics['test_rmse'].append(np.sqrt(mean_squared_error(y_test_fold, y_test_pred)))
		fold_metrics['train_mae'].append(mean_absolute_error(y_train_fold, y_train_pred))
		fold_metrics['test_mae'].append(mean_absolute_error(y_test_fold, y_test_pred))
		fold_metrics['train_r2'].append(r2_score(y_train_fold, y_train_pred))
		fold_metrics['test_r2'].append(r2_score(y_test_fold, y_test_pred))
		
		print(f"  Fold {fold_num + 1}: Test RMSE={fold_metrics['test_rmse'][-1]:.4f}, Test MAE={fold_metrics['test_mae'][-1]:.4f}, Test R²={fold_metrics['test_r2'][-1]:.4f}")
	
	X_poly = poly.fit_transform(X)
	X_poly_std = scaler.fit_transform(X_poly)
	regression_model.fit(X_poly_std, y)
	
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


def plot_error_by_true_bins(y_true_pred, bin_width=20, save_path=None, show: bool = True):
	if not y_true_pred:
		print("No predictions available to plot.")
		return None

	# Flatten 
	y_true_all = np.concatenate([yt for yt, _ in y_true_pred])
	y_pred_all = np.concatenate([yp for _, yp in y_true_pred])

	max_true = y_true_all.max()

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
		'bin_width': bin_width
	}


def main(metric: str = 'rmse', selected_features=None, alpha: float = 1.0, 
				 alpha_grid=None, n_splits=5, regularization='l2', polynomial_degrees=None, 
				 show_plots: bool = False, error_bin_width: int = 20):
	base_dir = Path(__file__).parent
	features_csv = base_dir.parent / "feature_data" / "segments_features.csv"

	# Default polynomial degrees if not specified
	if polynomial_degrees is None:
		polynomial_degrees = [1, 2, 3]

	print("=" * 70)
	print("Polynomial Regression for Segments Remaining Until Death (mean_speed)")
	print("=" * 70)

	print("\nLoading features and computing segments_from_end...")
	X, y, metadata, actual_features = load_and_prepare_data(features_csv, selected_features=selected_features)
	
	plot_mean_speed_vs_segments(metadata, save_path=base_dir / 'mean_speed_vs_segments.pdf', show=show_plots)
	
	plot_averaged_speed_vs_segments(metadata, window_size=5, save_path=base_dir / 'averaged_speed_vs_segments.pdf', show=show_plots)
	
	feature_names_used = actual_features
	
	if 'original_file' in metadata.columns:
		# Map worm names to numeric group indices
		worm_to_group = {worm: idx for idx, worm in enumerate(metadata['original_file'].unique())}
		groups = metadata['original_file'].map(worm_to_group).to_numpy()
		print(f"\nUsing GroupKFold with {len(worm_to_group)} groups (worms)")
	else:
		raise ValueError("Missing required column 'original_file' for GroupKFold")

	print(f"\nData Statistics:")
	print(f"  Features selected: {selected_features}")
	print(f"  X shape: {X.shape}, y shape: {y.shape}")
	print(f"  Segments from end - Min: {y.min():.2f}, Max: {y.max():.2f}, Mean: {y.mean():.2f}")

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

	# best degree based on chosen metric
	if metric.lower() == 'mae':
		best_pair = min(grid_results.keys(), key=lambda k: grid_results[k]['test_mae'])
		best_value = grid_results[best_pair]['test_mae']
	else:
		best_pair = min(grid_results.keys(), key=lambda k: grid_results[k]['test_rmse'])
		best_value = grid_results[best_pair]['test_rmse']

	print(f"\nBest (degree, alpha) by {metric.upper()}: {best_pair} (value: {best_value:.4f})")

	# Plot error by true-bin for the best model and print stats
	best_result = grid_results[best_pair]
	
	plot_error_by_true_bins(
		best_result['y_true_pred'],
		bin_width=error_bin_width,
		save_path=base_dir / 'error_by_true_bin.pdf'
	)

	# Compute RMSE/MAE 20-80 segments-from-death
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
	# configuration parameters
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
	
	ERROR_BIN_WIDTH = 5

	results = main(metric=SELECTED_METRIC, selected_features=SELECTED_FEATURES, alpha=ALPHA, 
			   alpha_grid=ALPHA_GRID, n_splits=N_SPLITS, regularization=REGULARIZATION,
		   polynomial_degrees=POLYNOMIAL_DEGREES, show_plots=SHOW_PLOTS,
		   error_bin_width=ERROR_BIN_WIDTH)
