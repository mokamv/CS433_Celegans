import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Optional, Tuple, Dict, List, Union, Any
from sklearn.model_selection import StratifiedKFold
import re


TIME_SERIES_FEATURES = ['x_coordinate', 'y_coordinate', 'speed', 'turning_angle']



def extract_features_and_labels(df):
    """Extract features and labels from dataframe."""
    metadata_columns = ['label', 'filename', 'relative_path', 'file', 'worm_id', 'segment_number', 'segment_index', 'original_file']
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    feature_columns = [col for col in numeric_columns if col not in metadata_columns]
    
    X = df[feature_columns].copy()
    y = df['label'].copy()
    
    # Extract base filename for proper grouping (segments from same worm grouped together)
    groups = df['filename'].apply(lambda x: extract_worm_and_segment_info(x)[0])
    
    return X, y, groups


def extract_worm_and_segment_info(filename):
    """Extract worm ID and segment number from filename."""
    # Handle segment files: filename-segment5.0-preprocessed.csv
    segment_match = re.search(r'segment(\d+)', filename)
    if segment_match:
        segment_num = int(segment_match.group(1))
        # Extract base filename without segment info
        worm_id = re.sub(r'-segment\d+.*', '', filename)
        return worm_id, segment_num
    
    # Handle full files or files without segment info
    worm_id = re.sub(r'-preprocessed.*', '', filename)
    return worm_id, 0


def create_kfold_splits(X, y, groups, n_splits=5):
    """Create file-based k-fold splits to prevent data leakage."""
    # Extract unique files and their labels
    file_df = pd.DataFrame({'file': groups, 'label': y}).drop_duplicates('file')
    unique_files = file_df['file'].values
    file_labels = file_df['label'].values
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_splits = []
    for train_file_idx, test_file_idx in skf.split(unique_files, file_labels):
        train_files = unique_files[train_file_idx]
        test_files = unique_files[test_file_idx]
        
        train_mask = groups.isin(train_files)
        test_mask = groups.isin(test_files)
        
        X_train = X[train_mask].reset_index(drop=True)
        X_test = X[test_mask].reset_index(drop=True)
        y_train = y[train_mask].reset_index(drop=True)
        y_test = y[test_mask].reset_index(drop=True)
        groups_train = groups[train_mask].reset_index(drop=True)
        groups_test = groups[test_mask].reset_index(drop=True)
        
        fold_splits.append({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'groups_train': groups_train,
            'groups_test': groups_test,
            'train_files': train_files,
            'test_files': test_files
        })
    
    return fold_splits



class LPBSDataLoader:
    """
    Comprehensive data loader for LPBS worm movement analysis.
    
    Features:
    - Load preprocessed trajectory data (time series) and extracted features
    - Support for both segment-level and full-trajectory data
    - File-based splitting to prevent data leakage
    - Easy filtering to first N segments per worm
    - Built-in cross-validation support
    - Data validation and statistics
    """
    
    def __init__(
        self,
        base_dir: str = ".",
        preprocessed_dir: str = "preprocessed_data",
        feature_dir: str = "feature_data"
    ):
        """
        Initialize the data loader.
        
        Args:
            base_dir: Base directory containing the project
            preprocessed_dir: Directory containing preprocessed trajectory data
            feature_dir: Directory containing extracted features
        """
        self.base_dir = Path(base_dir)
        self.preprocessed_dir = self.base_dir / preprocessed_dir
        self.feature_dir = self.base_dir / feature_dir
        
        self._segment_features = None
        self._full_features = None
        self._segment_timeseries = None
        self._full_timeseries = None
        self._metadata = {}
    
    def load_segment_features(self, force_reload: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Load segment-level features."""
        if self._segment_features is None or force_reload:
            df = pd.read_csv(self.feature_dir / "segments_features.csv")
            X, y, groups = extract_features_and_labels(df)
            self._segment_features = {'X': X, 'y': y, 'groups': groups}
            
        return self._segment_features['X'], self._segment_features['y'], self._segment_features['groups']
    
    
    def load_full_features(self, force_reload: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Load engineered features computed over full trajectories.

        Args:
            force_reload: Ignore cache and reload from disk when True.

        Returns:
            tuple: (X, y, groups) where X is a DataFrame of features, y are
            labels, and groups are worm-level identifiers for file-based CV.
        """
        if self._full_features is None or force_reload:
            df = pd.read_csv(self.feature_dir / "full_features.csv")
            X, y, groups = extract_features_and_labels(df)
            self._full_features = {'X': X, 'y': y, 'groups': groups}
            
        return self._full_features['X'], self._full_features['y'], self._full_features['groups']
    
    def load_segment_timeseries(self, force_reload: bool = False) -> Tuple[List, np.ndarray, np.ndarray]:
        """Load per-segment time series arrays and worm-level groups.

        Args:
            force_reload: Ignore cache and reload from disk when True.

        Returns:
            tuple: (X, y, groups) where X is a list of (T, 4) arrays with
            columns ['x','y','speed','turning_angle'], y are labels, and
            groups are worm-level identifiers extracted from filenames.
        """
        if self._segment_timeseries is None or force_reload:
            segments_dir = self.preprocessed_dir / "segments"
            metadata = pd.read_csv(segments_dir / "labels_and_metadata.csv")
            
            time_series_data = []
            labels = []
            groups = []
            
            for _, row in metadata.iterrows():
                try:
                    df = pd.read_csv(segments_dir / row['relative_path'] / row['file'])
                    ts_data = df[['x', 'y', 'speed', 'turning_angle']].fillna(0).values
                    time_series_data.append(ts_data)
                    labels.append(row['label'])
                    # Extract worm-level group (consistent with load_segment_features)
                    worm_id, _ = extract_worm_and_segment_info(row['file'])
                    groups.append(worm_id)
                except:
                    continue
            
            self._segment_timeseries = {
                'X': time_series_data,
                'y': np.array(labels),
                'groups': np.array(groups)
            }
            
        return self._segment_timeseries['X'], self._segment_timeseries['y'], self._segment_timeseries['groups']
    
    def load_full_timeseries(self, force_reload: bool = False) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """Load full-trajectory time series arrays and worm-level groups.

        Args:
            force_reload: Ignore cache and reload from disk when True.

        Returns:
            tuple: (X, y, groups) where X is a list of (T, 4) arrays with
            columns ['x','y','speed','turning_angle'], y are labels, and
            groups are worm-level identifiers extracted from filenames.
        """ 
        if self._full_timeseries is None or force_reload:
            full_dir = self.preprocessed_dir / "full"
            metadata = pd.read_csv(full_dir / 'labels_and_metadata.csv')
            
            time_series_data = []
            labels = []
            groups = []
            
            for _, row in metadata.iterrows():
                try:
                    df = pd.read_csv(full_dir / row['relative_path'] / row['file'])
                    ts_data = df[['x', 'y', 'speed', 'turning_angle']].fillna(0).values
                    time_series_data.append(ts_data)
                    labels.append(row['label'])
                    # Extract worm-level group (consistent with load_segment_features)
                    worm_id, _ = extract_worm_and_segment_info(row['file'])
                    groups.append(worm_id)
                except:
                    continue
            
            self._full_timeseries = {
                'X': time_series_data,
                'y': np.array(labels),
                'groups': np.array(groups)
            }
            
        return self._full_timeseries['X'], self._full_timeseries['y'], self._full_timeseries['groups']
    
    def create_cv_splits(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray], 
        groups: Union[pd.Series, np.ndarray],
        n_splits: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Create cross-validation splits with file-based grouping.
        
        Args:
            X: Feature matrix or time series data
            y: Labels
            groups: File identifiers for grouping
            n_splits: Number of CV splits
            
        Returns:
            List of fold split dictionaries
        """
        return create_kfold_splits(X, y, groups, n_splits)


if __name__ == "__main__":
    loader = LPBSDataLoader()
    
    X, y, groups = loader.load_segment_features()
    print(f"Loaded {X.shape[0]:,} samples with {X.shape[1]} features")
    
    X_first3, y_first3, groups_first3 = loader.get_first_n_segments(3, "features")
    cv_splits = loader.create_cv_splits(X_first3, y_first3, groups_first3, n_splits=3)
    print(f"Created {len(cv_splits)} CV splits")