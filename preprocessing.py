import numpy as np
import pandas as pd
import os
import re
import argparse
from tqdm import tqdm

# Constants
FRAME_RESET_VALUE = 10799
SEGMENT_LENGTH = 900
GAP_INTERPOLATION_LIMIT = 6
LONG_GAP_THRESHOLD = 7
EXCLUDED_FOLDERS = []
MAX_SPEED_THRESHOLD = 10.0


def cap_extreme_speeds(df, max_speed=MAX_SPEED_THRESHOLD):
    """Cap extreme speed values to prevent tracking artifacts from skewing analysis."""
    df = df.copy()
    df['speed'] = df['speed'].clip(upper=max_speed)
    return df

def calculate_turning_angle(df):
    """Compute per-frame turning angle in degrees.

    Args:
        df: DataFrame with at least `x` and `y` columns.

    Returns:
        pandas.DataFrame: Same dataframe with a `turning_angle` column in degrees
        added (first and last frames set to 0 if available).
    """
    if len(df) < 3 or 'x' not in df.columns or 'y' not in df.columns:
        df['turning_angle'] = 0
        return df
    
    dx = df['x'].diff().values
    dy = df['y'].diff().values
    
    angle = np.arctan2(dy, dx)
    angle_next = np.roll(angle, -1)
    turning_angle_rad = angle_next - angle
    turning_angle_rad = (turning_angle_rad + np.pi) % (2 * np.pi) - np.pi
    
    df['turning_angle'] = np.degrees(turning_angle_rad)
    df.loc[0, 'turning_angle'] = 0
    df.loc[df.index[-1], 'turning_angle'] = 0
    return df

def split_into_segments(df):
    """Split a trajectory dataframe into a list of per-segment dataframes.

    Expects a `segment` column marking segment membership.

    Args:
        df: DataFrame containing a `segment` column.

    Returns:
        list[pandas.DataFrame]: One dataframe per unique segment with a
        `segment_index` column added.
    """
    segments = []
    for segment_id in sorted(df['segment'].unique()):
        segment_df = df[df['segment'] == segment_id].copy()
        segment_df['segment_index'] = segment_id
        segments.append(segment_df)
    
    return segments

def clean_segment_gaps(segment_df):
    """Repair short gaps by interpolation and remove long gaps.

    Args:
        segment_df: DataFrame of a single segment containing columns `x`, `y`,
            and `speed` where gaps may be NaN.

    Returns:
        pandas.DataFrame: Cleaned segment dataframe with short gaps interpolated
        and rows from long gaps removed.
    """
    gap_mask = segment_df['x'].isna()
    is_nan = gap_mask.astype(int)
    starts = (is_nan.diff() == 1).astype(int)
    if len(is_nan) > 0 and is_nan.iloc[0] == 1:
        starts.iloc[0] = 1
        
    gap_ids = starts.cumsum() * is_nan
    
    rows_to_remove = []
    for gap_id in gap_ids[gap_ids > 0].unique():
        indices = segment_df.index[gap_ids == gap_id].tolist()
        gap_size = len(indices)
        
        if gap_size <= GAP_INTERPOLATION_LIMIT:
            if len(indices) > 0:
                start_idx = max(segment_df.index.min(), indices[0] - 1)
                end_idx = min(segment_df.index.max(), indices[-1] + 1)
                segment_df.loc[start_idx:end_idx, ['x', 'y', 'speed']] = segment_df.loc[start_idx:end_idx, ['x', 'y', 'speed']].interpolate(method='linear')
        elif gap_size >= LONG_GAP_THRESHOLD:
            rows_to_remove.extend(indices)

    if rows_to_remove:
        segment_df = segment_df.drop(index=rows_to_remove).reset_index(drop=True)
    
    return segment_df

def normalize_trajectory_data(df):
    """Normalize coordinates to [0,1] and turning angles to [-1,1]"""
    df_normalized = df.copy()
    
    # Global coordinate bounds from the dataset
    XY_MIN = 0.0
    XY_MAX = 749.0
    
    # Global turning angle bounds
    ANGLE_MIN = -180.0
    ANGLE_MAX = 180.0
    
    if 'x' in df.columns and 'y' in df.columns:
        df_normalized['x'] = (df['x'] - XY_MIN) / (XY_MAX - XY_MIN)
        df_normalized['y'] = (df['y'] - XY_MIN) / (XY_MAX - XY_MIN)
    
    if 'turning_angle' in df.columns:
        df_normalized['turning_angle'] = df['turning_angle'] / 180.0
    
    return df_normalized


_lifespan_data_cache = None

def load_lifespan_data(file='data/Lifespan/lifespan_summary.csv'):
    """Load and cache lifespan summary metadata.

    Returns:
        pandas.DataFrame: Cached dataframe loaded from file.
    """
    global _lifespan_data_cache
    if _lifespan_data_cache is None:
        _lifespan_data_cache = pd.read_csv(file)
        print(f"Loaded lifespan data: {len(_lifespan_data_cache)} files")
    return _lifespan_data_cache

def extract_file_pattern_from_path(file_path):
    """Extract the canonical filename pattern used in lifespan metadata.

    Args:
        file_path: Absolute or relative path to a raw CSV file.

    Returns:
        str | None: A pattern like `/<YYYYMMDD>_piworm<NN>_<recording>` if the
        filename matches the expected convention, otherwise None.
    """
    filename = os.path.basename(file_path)
    
    if 'coordinates_highestspeed_' in filename:
        match = re.search(r'coordinates_highestspeed_(\d{8})_(\d+)_(\d+)', filename)
        if match:
            date, worm_num, recording_num = match.groups()
            worm_num_padded = worm_num.zfill(2)
            piworm = f'piworm{worm_num_padded}'
            return f'/{date}_{piworm}_{recording_num}'
    
    return None

def preprocess_data(file_path, full_output_dir, segments_output_dir):
    """Preprocess a raw trajectory file into full and segmented outputs.

    Steps include capping speeds, computing turning angles, cleaning gaps,
    clipping to pre-death segments, normalization, and exporting CSVs.

    Args:
        file_path: Path to raw CSV input file.
        full_output_dir: Directory to write the full preprocessed CSV.
        segments_output_dir: Directory to write segment CSVs.

    Returns:
        tuple[list[tuple[str, int]] | None, list[tuple[str, int]] | None]:
        A pair of lists with (output_path, death_segment) for full and segment
        outputs respectively. Returns (None, None) if no lifespan match found.
    """
    # find matching line in lifespan_summary.csv
    lifespan_df = load_lifespan_data()
    file_pattern = extract_file_pattern_from_path(file_path)
    matching_rows = lifespan_df[lifespan_df['Filename'] == file_pattern]
    if len(matching_rows) == 0:
        print(f"No matching row found for {file_pattern}")
        return None, None
    
    #print(lifespan_df.columns.tolist())
    death_segment = int(matching_rows.iloc[0]['LifespanInFrames'] / SEGMENT_LENGTH) + 1 # +1 to include the death segment
    
    df_raw = pd.read_csv(file_path)
    #print(df_raw.columns.tolist())
    df_raw = df_raw.rename(columns={
        'GlobalFrame': 'frame', 'Speed': 'speed', 'X': 'x', 'Y': 'y'
    })
    df_raw = df_raw[['frame', 'speed', 'x', 'y']]
    df_raw = df_raw.iloc[1:].reset_index(drop=True) # remove first row because first frame is always very early compared to the second one
    
    df_raw = cap_extreme_speeds(df_raw)
    df_raw['frame'] = range(1, len(df_raw) + 1)
    df_raw['segment'] = (df_raw['frame'] - 1) // SEGMENT_LENGTH
    
    df_raw = df_raw[df_raw['segment'] < death_segment]  # remove segments after death
    
    segments = split_into_segments(df_raw)
    
    cleaned_segments = []
    for segment_df in segments:
        segment_cleaned = clean_segment_gaps(segment_df)
        segment_cleaned = calculate_turning_angle(segment_cleaned)
        cleaned_segments.append(segment_cleaned)
    
    df_cleaned = pd.concat(cleaned_segments, ignore_index=True)
    
    all_full_results = []
    all_segments_results = []
    
    # Save full dataset
    df_full = normalize_trajectory_data(df_cleaned)
    full_preprocessed_filename = os.path.join(full_output_dir, os.path.basename(file_path).replace(".csv", "-preprocessed.csv"))
    df_full.to_csv(full_preprocessed_filename, index=False)
    all_full_results.append((full_preprocessed_filename, death_segment))
    
    # Save segments
    segments = split_into_segments(df_cleaned)
    
    for segment_df in segments:
        if len(segment_df) == 0:
            continue
            
        segment_index = segment_df['segment_index'].iloc[0]
        segment_df = normalize_trajectory_data(segment_df)
        
        segment_filename = os.path.basename(file_path).replace(".csv", f"-segment{segment_index}-preprocessed.csv")
        segment_path = os.path.join(segments_output_dir, segment_filename)
        
        segment_df.to_csv(segment_path, index=False)
        all_segments_results.append((segment_path, death_segment))
    
    return all_full_results, all_segments_results


def process_directory(input_dir, output_dir):
    """Preprocess all raw CSVs under a directory tree.

    Traverses subdirectories (excluding some), preprocesses each file into
    full and segmented CSVs, and writes `labels_and_metadata.csv` for both.

    Args:
        input_dir: Base directory containing raw CSV files under treatments.
        output_dir: Base output directory for `full/` and `segments/` results.

    Returns:
        None
    """
    segments_output_dir = os.path.join(output_dir, "segments")
    full_output_dir = os.path.join(output_dir, "full")
    os.makedirs(segments_output_dir, exist_ok=True)
    os.makedirs(full_output_dir, exist_ok=True)
    
    file_labels = {}
    processing_args = []
    
    print(f"Scanning directory: {input_dir} for CSV files...")
    for root, dirs, files in os.walk(input_dir):
        path_components = os.path.normpath(root).split(os.sep)
        if any(excluded_folder in path_components for excluded_folder in EXCLUDED_FOLDERS):
            continue
            
        for filename in files:
            if filename.endswith(".csv") and not any(skip in filename.lower() for skip in ['summary', 'metadata', 'labels']):
                file_path = os.path.join(root, filename)
                label = 1 if "+" in root else 0
                rel_path = os.path.relpath(os.path.dirname(file_path), input_dir)
            
                segments_rel_dir = os.path.join(segments_output_dir, rel_path)
                full_rel_dir = os.path.join(full_output_dir, rel_path)
                os.makedirs(segments_rel_dir, exist_ok=True)
                os.makedirs(full_rel_dir, exist_ok=True)
            
                processing_args.append((file_path, full_rel_dir, segments_rel_dir, filename, label, rel_path))
                file_labels[filename] = label

    total_files = len(processing_args)
    print(f"Found {total_files} CSV files to process")
    
    segments_metadata = []
    full_metadata = []    
    for file_path, full_rel_dir, segments_rel_dir, filename, label, rel_path in tqdm(processing_args, desc="Processing"):
        
        full_results, segments_results = preprocess_data(file_path, full_rel_dir, segments_rel_dir)
        if full_results is None or segments_results is None:
            continue
        
        for full_result in full_results:
            if full_result[0] is not None:
                preprocessed_file, death_segment = full_result
                preprocessed_basename = os.path.basename(preprocessed_file)
                
                full_metadata.append({
                    'file': preprocessed_basename,
                    'original_file': filename,
                    'death_segment': death_segment,
                    'label': label,
                    'relative_path': rel_path
                })
            
        for segment_path, death_segment_value in segments_results:
            if segment_path is not None:
                segment_basename = os.path.basename(segment_path)
                segment_match = re.search(r'-segment(\d+)-', segment_basename)
                segment_index = int(segment_match.group(1)) if segment_match else None
                    
                segments_metadata.append({
                    'file': segment_basename,
                    'original_file': filename,
                    'segment_index': segment_index,
                    'death_segment': death_segment_value,
                    'label': label,
                    'relative_path': rel_path
                })
    
    segments_df = pd.DataFrame(segments_metadata)
    segments_df.to_csv(os.path.join(segments_output_dir, "labels_and_metadata.csv"), index=False)
        
    label_counts = segments_df['label'].value_counts()
    print(f"\nSegments: {dict(label_counts)}")
    
    if full_metadata:
        full_df = pd.DataFrame(full_metadata)
        full_df.to_csv(os.path.join(full_output_dir, "labels_and_metadata.csv"), index=False)
        
        label_counts = full_df['label'].value_counts()
        print(f"Full files: {dict(label_counts)}")
    
    print(f"\n Processing completed")
    print(f"Full files: {len(full_metadata)}, Segments: {len(segments_metadata)}")

if __name__ == "__main__":
    input_dir = "data"
    output_dir = "preprocessed_data"
    
    process_directory(input_dir, output_dir)