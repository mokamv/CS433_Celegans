from scipy.stats import median_abs_deviation, iqr
import os
import warnings

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import entropy, skew, kurtosis
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft

import pywt
from tqdm import tqdm

# Suppress PyWavelets warnings for short signals
warnings.filterwarnings('ignore', category=UserWarning, module='pywt')

def calculate_enhanced_physics_features(df):
    """Calculate physics-inspired movement features."""
    features = {}
    
    # Drop NaN values before calculating features
    speed = df['speed'].dropna()
    turning = df['turning_angle'].dropna()
    
    if len(speed) == 0:
        features.update({
            'kinetic_energy_proxy': 0,
            'movement_efficiency': 0,
            'speed_persistence': 0,
            'exploration_ratio': 0
        })
        return features
    
    # Kinetic energy proxy (mass assumed constant)
    features['kinetic_energy_proxy'] = np.mean(0.5 * speed**2)
    
    # Movement efficiency
    max_speed = speed.max() if len(speed) > 0 else 1
    features['movement_efficiency'] = speed.sum() / (max_speed * len(speed) + 1e-6)
    
    # Speed persistence (consistency)
    speed_std = speed.std()
    if np.isnan(speed_std) or speed_std == 0:
        features['speed_persistence'] = 0
    else:
        features['speed_persistence'] = speed.mean() / (speed_std + 1e-6)
    
    # Exploration ratio (turning vs speed)
    if len(turning) > 0:
        turning_freq = np.sum(np.abs(turning) > 30) / len(turning)  # Significant turns
        features['exploration_ratio'] = turning_freq / (speed.mean() + 1e-6)
    else:
        features['exploration_ratio'] = 0
    
    return features

def calculate_enhanced_statistical_features(df):
    """Calculate advanced statistical features."""
    features = {}
    
    # Drop NaN values before calculating features
    speed = df['speed'].dropna()
    turning = df['turning_angle'].dropna()
    
    if len(speed) == 0:
        features.update({
            'speed_skewness': 0,
            'speed_kurtosis': 0,
            'speed_iqr': 0,
            'speed_cv': 0
        })
        return features
    
    # Skewness and kurtosis for speed
    if len(speed) > 2:
        speed_std = speed.std()
        if speed_std == 0 or np.isnan(speed_std):
            # When all values are identical (variance = 0), set skew/kurtosis to 0
            features['speed_skewness'] = 0
            features['speed_kurtosis'] = 0
        else:
            skew_val = skew(speed)
            kurt_val = kurtosis(speed)
            # Handle any remaining NaN cases
            features['speed_skewness'] = 0 if np.isnan(skew_val) else skew_val
            features['speed_kurtosis'] = 0 if np.isnan(kurt_val) else kurt_val
    else:
        features['speed_skewness'] = 0
        features['speed_kurtosis'] = 0
    
    # Interquartile range
    features['speed_iqr'] = np.percentile(speed, 75) - np.percentile(speed, 25)
    
    # Coefficient of variation
    speed_std = speed.std()
    speed_mean = speed.mean()
    if np.isnan(speed_std) or speed_std == 0 or np.isnan(speed_mean):
        features['speed_cv'] = 0
    else:
        features['speed_cv'] = speed_std / (speed_mean + 1e-6)
    
    return features

def calculate_enhanced_temporal_features(df):
    """Calculate temporal pattern features."""
    features = {}
    
    # Drop NaN values before calculating features
    speed = df['speed'].dropna()
    
    if len(speed) < 3:
        features.update({
            'speed_trend': 0,
            'speed_acceleration_mean': 0,
            'speed_acceleration_std': 0
        })
        return features
    
    # Speed trend (linear fit slope)
    try:
        features['speed_trend'] = np.polyfit(range(len(speed)), speed, 1)[0]
    except:
        features['speed_trend'] = 0
    
    # Speed acceleration (rate of change)
    speed_accel = np.diff(speed)
    features['speed_acceleration_mean'] = np.mean(speed_accel)
    features['speed_acceleration_std'] = np.std(speed_accel)
    
    return features

def calculate_enhanced_frequency_features(df):
    """Calculate frequency domain features using FFT."""
    features = {}
    
    # Drop NaN values before calculating features
    speed = df['speed'].dropna()
    turning = df['turning_angle'].dropna()
    
    def get_dominant_frequency(x):
        if len(x) < 8:  # Need minimum points for meaningful FFT
            return 0
        try:
            fft_vals = np.abs(fft(x))[1:len(x)//2]
            if len(fft_vals) == 0:
                return 0
            return np.argmax(fft_vals)
        except:
            return 0
    
    def get_spectral_centroid(x):
        if len(x) < 8:
            return 0
        try:
            fft_vals = np.abs(fft(x))[1:len(x)//2]
            if len(fft_vals) == 0:
                return 0
            freqs = np.arange(1, len(fft_vals) + 1)
            return np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-6)
        except:
            return 0
    
    # Frequency analysis of speed
    features['speed_dominant_freq'] = get_dominant_frequency(speed)
    features['speed_spectral_centroid'] = get_spectral_centroid(speed)
    
    # Frequency analysis of turning
    features['turning_dominant_freq'] = get_dominant_frequency(turning)
    features['turning_spectral_centroid'] = get_spectral_centroid(turning)
    
    return features

def calculate_behavioral_state_features(df):
    """Calculate features representing different behavioral states."""
    features = {}
    
    # Drop NaN values before calculating features
    speed = df['speed'].dropna()
    turning = df['turning_angle'].dropna()
    
    if len(speed) == 0 or len(turning) == 0:
        features.update({
            'activity_level': 0,
            'high_activity_fraction': 0,
            'low_activity_fraction': 0,
            'mixed_activity_fraction': 0
        })
        return features
    
    # Calculate activity level combining speed and turning
    turning_activity = np.abs(turning)
    activity_scores = []
    
    min_len = min(len(speed), len(turning_activity))
    if min_len > 0:
        speed_norm = speed[:min_len] / (speed.max() + 1e-6)
        turning_norm = turning_activity[:min_len] / (turning_activity.max() + 1e-6)
        activity_scores = speed_norm * 0.7 + turning_norm * 0.3  # Weight speed more
    
    if len(activity_scores) > 0:
        activity_mean = np.mean(activity_scores)
        features['activity_level'] = 0 if np.isnan(activity_mean) else activity_mean
        
        # Activity state fractions
        high_threshold = np.percentile(activity_scores, 75)
        low_threshold = np.percentile(activity_scores, 25)
        
        features['high_activity_fraction'] = np.mean(activity_scores > high_threshold)
        features['low_activity_fraction'] = np.mean(activity_scores < low_threshold)
        features['mixed_activity_fraction'] = np.mean((activity_scores >= low_threshold) & 
                                                    (activity_scores <= high_threshold))
    else:
        features['activity_level'] = 0
        features['high_activity_fraction'] = 0
        features['low_activity_fraction'] = 0
        features['mixed_activity_fraction'] = 0
    
    return features


def calculate_basic_features(df, speed_threshold=0.05):
    """Calculate basic movement features.
    
    Args:
        df: DataFrame containing movement data
        speed_threshold: Minimum speed to consider as movement
    """
    features = {}
    
    # Drop NaN values before calculating features
    speed = df['speed'].dropna()
    x = df['x'].dropna()
    y = df['y'].dropna()
    
    if len(speed) == 0 or len(x) == 0 or len(y) == 0:
        features.update({
            'mean_speed': 0,
            'std_speed': 0,
            'max_speed': 0,
            'min_speed': 0,
            'total_distance': 0,
            'time_paused': 0,
            'fraction_paused': 0
        })
        return features
    
    # Speed features
    features['mean_speed'] = speed.mean()
    std_speed = speed.std()
    features['std_speed'] = 0 if np.isnan(std_speed) else std_speed
    features['max_speed'] = speed.max()
    features['min_speed'] = speed.min()
    
    # Total distance traveled
    dx = x.diff()
    dy = y.diff()
    distances = np.sqrt(dx**2 + dy**2)
    features['total_distance'] = distances.sum()
    
    # Time paused (speed below threshold)
    features['time_paused'] = (speed < speed_threshold).sum()
    features['fraction_paused'] = features['time_paused'] / len(speed)
    
    return features

def calculate_turning_features(df, window_size=30, height_threshold=90):
    """Calculate features related to turning behavior.
    
    Args:
        df: DataFrame containing movement data
        window_size: Size of windows for calculating meandering ratio
        height_threshold: Minimum height for peaks in turning angle
    """
    features = {}
    
    # Drop NaN values before calculating features
    turning_angle = df['turning_angle'].dropna()
    x = df['x'].dropna()
    y = df['y'].dropna()
    
    if len(turning_angle) == 0 or len(x) == 0 or len(y) == 0:
        features.update({
            'mean_turning_angle': 0,
            'std_turning_angle': 0,
            'max_turning_angle': 0,
            'turning_frequency': 0,
            'mean_meandering_ratio': 0,
            'std_meandering_ratio': 0,
            'min_meandering_ratio': 0,
            'max_meandering_ratio': 0,
            'median_meandering_ratio': 0,
            'fraction_efficient_movement': 0,
        })
        return features
    
    # Turning angle statistics
    features['mean_turning_angle'] = turning_angle.mean()
    std_turning = turning_angle.std()
    features['std_turning_angle'] = 0 if np.isnan(std_turning) else std_turning
    features['max_turning_angle'] = turning_angle.abs().max()
    
    # Turning frequency (peaks in turning angle)
    peaks, _ = find_peaks(turning_angle.abs(), height=height_threshold)
    features['turning_frequency'] = len(peaks) / len(turning_angle)
    
    # Calculate meandering ratio in non-overlapping windows
    meandering_ratios = []
    for i in range(0, len(x), window_size):
        if i + window_size > len(x):
            continue
            
        window_x = x.iloc[i:i+window_size]
        window_y = y.iloc[i:i+window_size]
        
        net_displacement = np.sqrt((window_x.iloc[-1] - window_x.iloc[0])**2 + 
                                 (window_y.iloc[-1] - window_y.iloc[0])**2)
        
        dx = window_x.diff()
        dy = window_y.diff()
        total_distance = np.sqrt(dx**2 + dy**2).sum()
        
        if total_distance > 0:
            meandering_ratios.append(net_displacement / total_distance)
    
    if meandering_ratios:
        features['mean_meandering_ratio'] = np.mean(meandering_ratios)
        features['std_meandering_ratio'] = np.std(meandering_ratios)
        features['min_meandering_ratio'] = np.min(meandering_ratios)
        features['max_meandering_ratio'] = np.max(meandering_ratios)
        features['median_meandering_ratio'] = np.median(meandering_ratios)
        features['fraction_efficient_movement'] = np.mean(np.array(meandering_ratios) > 0.5)
    else:
        features['mean_meandering_ratio'] = 0
        features['std_meandering_ratio'] = 0
        features['min_meandering_ratio'] = 0
        features['max_meandering_ratio'] = 0
        features['median_meandering_ratio'] = 0
        features['fraction_efficient_movement'] = 0
    
    return features

def calculate_higuchi_fd(x, k_max=10):
    """Calculate the Higuchi fractal dimension of a 1D signal.

    Args:
        x: 1D array-like signal.
        k_max: Maximum k for the Higuchi algorithm (windowing scale).

    Returns:
        float: Estimated fractal dimension in [0, 2] (0 when insufficient
        data or if computation fails).
    """
    try:
        n = len(x)
        if n < 2 * k_max:  # Need sufficient data points
            return 0
            
        l = []
        for k in range(1, k_max + 1):
            lk = 0
            for m in range(k):
                lkm = 0
                num_points = (n - m) // k
                if num_points <= 1:  # Need at least 2 points for calculation
                    continue
                    
                for i in range(1, num_points):
                    lkm += abs(x[m + i * k] - x[m + (i - 1) * k])
                
                if num_points > 0 and k > 0:
                    lkm = lkm * (n - 1) / (num_points * k)
                    lk += lkm
            
            # Avoid log of zero or negative values
            if lk > 0 and k > 0:
                l.append(np.log(lk / k))
            else:
                return 0  # Return 0 if we encounter invalid values
        
        if len(l) < 2:  # Need at least 2 points for polyfit
            return 0
            
        hfd = np.polyfit(np.log(range(1, len(l) + 1)), l, 1)[0]
        return hfd
    except Exception as e:
        return 0

def calculate_entropy_features(df):
    """Calculate entropy and randomness measures."""
    features = {}
    
    # Drop NaN values before calculating features
    speed = df['speed'].dropna().values
    turning = df['turning_angle'].dropna().values
    
    # Only calculate entropy if we have enough non-NaN values
    if len(speed) > 0:
        features['speed_entropy'] = entropy(np.histogram(speed, bins=20)[0])
        features['speed_fractal_dim'] = calculate_higuchi_fd(speed)
    else:
        features['speed_entropy'] = 0
        features['speed_fractal_dim'] = 0
    
    if len(turning) > 0:
        features['turning_entropy'] = entropy(np.histogram(turning, bins=20)[0])
        features['turning_fractal_dim'] = calculate_higuchi_fd(turning)
    else:
        features['turning_entropy'] = 0
        features['turning_fractal_dim'] = 0
    
    return features

def calculate_wavelet_features(df, max_level=3):
    """Calculate features using wavelet analysis with dynamic level assignment."""
    features = {}
    
    # Drop NaN values before calculating features
    speed = df['speed'].dropna().values
    turning = df['turning_angle'].dropna().values
    
    def get_safe_level(signal_length, max_level):
        """Calculate the maximum safe decomposition level for a given signal length."""
        if signal_length < 4:  # Need minimum 4 points for any wavelet decomposition
            return 0
        # Rule: signal length should be at least 2^level, with buffer for boundary effects
        safe_level = min(max_level, int(np.log2(signal_length)) - 1)
        return max(0, safe_level)  # Can be 0 for very short signals
    
    # Initialize all possible levels to 0
    for i in range(max_level + 1):
        features[f'wavelet_speed_level{i}'] = 0
        features[f'wavelet_turning_level{i}'] = 0
    
    # Calculate wavelet features for speed
    if len(speed) > 0:
        speed_level = get_safe_level(len(speed), max_level)
        if speed_level > 0:
            try:
                coeffs = pywt.wavedec(speed, 'db4', level=speed_level)
                for i, c in enumerate(coeffs):
                    if i <= max_level:  # Ensure we don't exceed our feature space
                        features[f'wavelet_speed_level{i}'] = np.std(c) if len(c) > 0 else 0
            except Exception:
                # If wavelet decomposition fails, features remain 0
                pass
    
    # Calculate wavelet features for turning
    if len(turning) > 0:
        turning_level = get_safe_level(len(turning), max_level)
        if turning_level > 0:
            try:
                coeffs = pywt.wavedec(turning, 'db4', level=turning_level)
                for i, c in enumerate(coeffs):
                    if i <= max_level:  # Ensure we don't exceed our feature space
                        features[f'wavelet_turning_level{i}'] = np.std(c) if len(c) > 0 else 0
            except Exception:
                # If wavelet decomposition fails, features remain 0
                pass
    
    return features

def calculate_roaming_dwelling_score(df, window_size=30):
    """Calculate roaming vs dwelling score."""
    # Early exit for insufficient data
    if len(df) < window_size or 'speed' not in df.columns or 'turning_angle' not in df.columns:
        return {
            'mean_roaming_score': 0,
            'std_roaming_score': 0,
            'fraction_roaming': 0
        }
    
    # Drop NaN values and convert to numpy arrays for faster processing
    speed = df['speed'].dropna().values
    turning = df['turning_angle'].dropna().values
    
    if len(speed) < window_size or len(turning) < window_size:
        return {
            'mean_roaming_score': 0,
            'std_roaming_score': 0,
            'fraction_roaming': 0
        }
    
    # Initialize result dictionary
    features = {
        'mean_roaming_score': 0,
        'std_roaming_score': 0,
        'fraction_roaming': 0
    }
    
    try:
        # Apply gaussian filter for smoothing
        speed_smooth = gaussian_filter1d(speed, sigma=3)
        turning_smooth = gaussian_filter1d(turning, sigma=3)
        
        # Calculate scores in non-overlapping windows
        roaming_scores = []
        
        for i in range(0, len(speed_smooth) - window_size + 1, window_size):
            if i + window_size <= len(speed_smooth) and i + window_size <= len(turning_smooth):
                speed_window = speed_smooth[i:i+window_size]
                turning_window = turning_smooth[i:i+window_size]
                
                # Skip windows with all zeros or NaNs
                if np.all(np.isnan(speed_window)) or np.all(np.isnan(turning_window)):
                    continue
                
                # Replace NaNs with zeros to avoid calculation errors
                speed_window = np.nan_to_num(speed_window, nan=0.0)
                turning_window = np.nan_to_num(turning_window, nan=0.0)
                
                # Calculate mean speed and turning scores
                speed_score = np.mean(speed_window)
                turning_score = np.mean(np.abs(turning_window))
                
                # Calculate roaming score
                # High speed and low turning = roaming (positive score)
                # Low speed and high turning = dwelling (negative score)
                score = (speed_score - turning_score/180) / 2
                roaming_scores.append(score)
        
        # Calculate features if we have valid scores
        if roaming_scores:
            roaming_scores_array = np.array(roaming_scores)
            features['mean_roaming_score'] = np.mean(roaming_scores_array)
            features['std_roaming_score'] = np.std(roaming_scores_array)
            features['fraction_roaming'] = np.mean(roaming_scores_array > 0)
    except Exception as e:
        # In case of any error, return zeros
        print(f"Error in calculate_roaming_dwelling_score: {e}")
    
    return features

def calculate_frenetic_movement(df, window_size=10, overlap=5):
    """Calculate metrics for short-term erratic/frenetic movement.
    
    Args:
        df: DataFrame containing movement data
        window_size: Size of windows for calculating frenetic movement (smaller than other functions)
        overlap: Overlap between consecutive windows
    """
    # Fast path for empty or insufficient data
    if len(df) < window_size or 'speed' not in df.columns or 'turning_angle' not in df.columns:
        return {
            'mean_frenetic_score': 0,
            'max_frenetic_score': 0,
            'std_frenetic_score': 0,
            'pct_high_frenetic': 0,
            'mean_jerk': 0,
            'max_jerk': 0
        }
    
    # Drop NaN values and convert to numpy arrays for faster processing
    speed = df['speed'].dropna().values
    turning = df['turning_angle'].dropna().values
    
    if len(speed) < window_size or len(turning) < window_size:
        return {
            'mean_frenetic_score': 0,
            'max_frenetic_score': 0,
            'std_frenetic_score': 0,
            'pct_high_frenetic': 0,
            'mean_jerk': 0,
            'max_jerk': 0
        }
    
    # Calculate metrics in sliding windows
    frenetic_scores = []
    jerk_values = []
    
    # Pre-compute step size
    step = window_size - overlap
    
    # Pre-allocate temporary arrays to avoid repeated allocations
    windows_count = max(1, (len(speed) - window_size) // step + 1)
    
    try:
        for i in range(0, len(speed) - window_size + 1, step):
            window_speed = speed[i:i+window_size]
            window_turning = turning[i:i+window_size]
            
            # Calculate speed acceleration
            speed_accel = np.diff(window_speed)
            
            # Calculate jerk (rate of change of acceleration)
            if len(speed_accel) > 1:
                jerk = np.diff(speed_accel)
                if len(jerk) > 0:
                    mean_jerk = np.mean(np.abs(jerk))
                    if not np.isnan(mean_jerk):
                        jerk_values.append(mean_jerk)
            
            # Calculate turning acceleration
            turning_accel = np.abs(np.diff(window_turning))
            
            # Calculate frenetic score
            if len(speed_accel) > 1 and len(turning_accel) > 1:
                speed_std = np.std(speed_accel)
                turning_std = np.std(turning_accel)
                
                if not np.isnan(speed_std) and not np.isnan(turning_std) and speed_std > 0 and turning_std > 0:
                    frenetic_score = speed_std * turning_std
                    frenetic_scores.append(frenetic_score)
    except Exception:
        # If any calculation fails, return zeros
        return {
            'mean_frenetic_score': 0,
            'max_frenetic_score': 0,
            'std_frenetic_score': 0,
            'pct_high_frenetic': 0,
            'mean_jerk': 0,
            'max_jerk': 0
        }
    
    # Initialize results
    features = {
        'mean_frenetic_score': 0,
        'max_frenetic_score': 0,
        'std_frenetic_score': 0,
        'pct_high_frenetic': 0,
        'mean_jerk': 0,
        'max_jerk': 0
    }
    
    # Calculate overall features if we have valid scores
    if frenetic_scores:
        # Convert to numpy array once for faster calculations
        frenetic_scores_array = np.array(frenetic_scores)
        features['mean_frenetic_score'] = np.mean(frenetic_scores_array)
        features['max_frenetic_score'] = np.max(frenetic_scores_array)
        features['std_frenetic_score'] = np.std(frenetic_scores_array)
        
        # Safely calculate percentile
        if len(frenetic_scores) >= 4:
            percentile_75 = np.percentile(frenetic_scores_array, 75)
            features['pct_high_frenetic'] = np.mean(frenetic_scores_array > percentile_75)
    
    # Calculate jerk features if we have valid values
    if jerk_values:
        jerk_values_array = np.array(jerk_values)
        features['mean_jerk'] = np.mean(jerk_values_array)
        features['max_jerk'] = np.max(jerk_values_array)
    
    return features

def calculate_additional_features(df):
    """Calculate some additional advanced features for richer analysis.
    Added by Joost."""
    features = {}
   # Median absolute deviation (MAD) — safe
    for col in ['speed', 'x', 'y', 'turning_angle']:
        if col in df.columns:
            v = df[col].dropna().values
            if len(v) > 0:
                mad = median_abs_deviation(v, scale=1)
                features[f'{col}_mad'] = 0 if np.isnan(mad) else mad
            else:
                features[f'{col}_mad'] = 0

    # Interquartile range (IQR)
    for col in ['x', 'y', 'turning_angle']:
        if col in df.columns:
            features[f'{col}_iqr'] = iqr(df[col].dropna())
    # Net displacement and straightness index
    # Net displacement and straightness index (robust)
    if set(['x', 'y']).issubset(df.columns):
        xy = df[['x', 'y']].dropna()
        if len(xy) > 1:
            x = xy['x'].values
            y = xy['y'].values

            net_disp = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
            step_dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
            total_dist = np.sum(step_dist)

            features['net_displacement_all'] = net_disp
            features['straightness_index_all'] = (
                net_disp / total_dist if total_dist > 0 else 0
            )
        else:
            features['net_displacement_all'] = 0
            features['straightness_index_all'] = 0

    # Time to max speed
    if 'speed' in df.columns and 'time' in df.columns:
        speed = df['speed'].dropna()
        time = df['time'].dropna()
        if len(speed) > 0 and len(time) > 0:
            idx = np.argmax(speed)
            features['time_to_max_speed'] = time.iloc[idx] - time.iloc[0]
    # Acceleration and jerk statistics (robust)
    if 'speed' in df.columns:
        speed = df['speed'].dropna().values

        if len(speed) > 2:
            accel = np.diff(speed)
            accel = accel[np.isfinite(accel)]

            if len(accel) > 0:
                features['acceleration_median'] = np.median(accel)
                features['acceleration_iqr'] = iqr(accel) if len(accel) > 1 else 0
            else:
                features['acceleration_median'] = 0
                features['acceleration_iqr'] = 0

            jerk = np.diff(accel)
            jerk = jerk[np.isfinite(jerk)]

            if len(jerk) > 0:
                features['jerk_median'] = np.median(jerk)
                features['jerk_iqr'] = iqr(jerk) if len(jerk) > 1 else 0
            else:
                features['jerk_median'] = 0
                features['jerk_iqr'] = 0
        else:
            features['acceleration_median'] = 0
            features['acceleration_iqr'] = 0
            features['jerk_median'] = 0
            features['jerk_iqr'] = 0
    return features

def calculate_state_features(
    df,
    speed_eps=0.05,
    turn_thresh=30
):
    """
    Discretize behavior into PAUSE / RUN / TURN states and extract
    state occupancy, run/pause length statistics, and transition dynamics.
    Added by Joost."""
    features = {}

    if 'speed' not in df.columns or 'turning_angle' not in df.columns:
        return {k: 0 for k in [
            'frac_pause','frac_run','frac_turn',
            'mean_run_length','mean_pause_length',
            'max_pause_length','run_pause_ratio',
            'transition_entropy'
        ]}

    speed = df['speed'].dropna().values
    turning = df['turning_angle'].dropna().values
    n = min(len(speed), len(turning))
    if n < 5:
        return {k: 0 for k in [
            'frac_pause','frac_run','frac_turn',
            'mean_run_length','mean_pause_length',
            'max_pause_length','run_pause_ratio',
            'transition_entropy'
        ]}

    speed = speed[:n]
    turning = turning[:n]

    # States: 0 = pause, 1 = run, 2 = turn
    states = np.zeros(n, dtype=int)
    states[speed < speed_eps] = 0
    states[(speed >= speed_eps) & (np.abs(turning) < turn_thresh)] = 1
    states[np.abs(turning) >= turn_thresh] = 2

    # Fractions
    features['frac_pause'] = np.mean(states == 0)
    features['frac_run']   = np.mean(states == 1)
    features['frac_turn']  = np.mean(states == 2)

    # Run-length encoding
    lengths = []
    current = states[0]
    count = 1
    for s in states[1:]:
        if s == current:
            count += 1
        else:
            lengths.append((current, count))
            current = s
            count = 1
    lengths.append((current, count))

    run_lengths   = [l for s,l in lengths if s == 1]
    pause_lengths = [l for s,l in lengths if s == 0]

    features['mean_run_length']   = np.mean(run_lengths) if run_lengths else 0
    features['mean_pause_length'] = np.mean(pause_lengths) if pause_lengths else 0
    features['max_pause_length']  = np.max(pause_lengths) if pause_lengths else 0
    features['run_pause_ratio']   = (
        np.sum(states == 1) / (np.sum(states == 0) + 1e-6)
    )

    # Transition matrix entropy
    T = np.zeros((3,3))
    for i in range(len(states)-1):
        T[states[i], states[i+1]] += 1
    T = T / (T.sum() + 1e-6)
    features['transition_entropy'] = entropy(T.flatten() + 1e-12)

    return features

def calculate_burst_features(df, speed_eps=0.05):
    """Calculate burst movement features.
    Added by Joost."""
    features = {}

    if 'speed' not in df.columns:
        return {
            'burst_count':0,
            'mean_burst_length':0,
            'burst_duty_cycle':0,
            'interburst_cv':0
        }

    speed = df['speed'].dropna().values
    if len(speed) < 5:
        return {
            'burst_count':0,
            'mean_burst_length':0,
            'burst_duty_cycle':0,
            'interburst_cv':0
        }

    active = speed > speed_eps

    bursts = []
    current = 0
    for a in active:
        if a:
            current += 1
        else:
            if current > 0:
                bursts.append(current)
                current = 0
    if current > 0:
        bursts.append(current)

    interburst = np.diff(np.where(active)[0]) if np.any(active) else []

    features['burst_count'] = len(bursts)
    features['mean_burst_length'] = np.mean(bursts) if bursts else 0
    features['burst_duty_cycle'] = np.mean(active)
    features['interburst_cv'] = (
        np.std(interburst) / (np.mean(interburst) + 1e-6)
        if len(interburst) > 1 else 0
    )

    return features

def calculate_degradation_features(df, speed_eps=0.05):
    """Calculate features related to movement degradation over time.
    Added by Joost."""
    features = {}

    if 'speed' not in df.columns:
        return {
            'early_late_speed_ratio':0,
            'pause_increase':0
        }

    speed = df['speed'].dropna().values
    if len(speed) < 10:
        return {
            'early_late_speed_ratio':0,
            'pause_increase':0
        }

    q = len(speed) // 4
    early = speed[:q]
    late  = speed[-q:]

    features['early_late_speed_ratio'] = (
        np.mean(early) / (np.mean(late) + 1e-6)
        if np.mean(late) > 0 else 0
    )

    early_pause = np.mean(early < speed_eps)
    late_pause  = np.mean(late  < speed_eps)
    features['pause_increase'] = late_pause - early_pause

    return features

def calculate_turn_event_features(df, turn_thresh=30):
    """Calculate features related to discrete turn events.
    Added by Joost."""
    features = {}

    if 'turning_angle' not in df.columns:
        return {
            'turn_event_rate':0,
            'turn_clustering':0
        }

    turning = np.abs(df['turning_angle'].dropna().values)
    if len(turning) < 5:
        return {
            'turn_event_rate':0,
            'turn_clustering':0
        }

    events = np.where(turning > turn_thresh)[0]
    features['turn_event_rate'] = len(events) / len(turning)

    if len(events) > 2:
        intervals = np.diff(events)
        features['turn_clustering'] = np.std(intervals) / (np.mean(intervals) + 1e-6)
    else:
        features['turn_clustering'] = 0

    return features

def calculate_terminal_features(df, speed_eps=0.05):
    """Calculate features related to terminal movement behavior.
    Added by Joost."""
    features = {}

    if 'speed' not in df.columns:
        return {
            'last_active_frame':0,
            'terminal_pause_fraction':0
        }

    speed = df['speed'].dropna().values
    n = len(speed)
    if n < 5:
        return {
            'last_active_frame':0,
            'terminal_pause_fraction':0
        }

    active_idx = np.where(speed > speed_eps)[0]
    features['last_active_frame'] = active_idx[-1] if len(active_idx) else 0

    tail = speed[int(0.9*n):]
    features['terminal_pause_fraction'] = np.mean(tail < speed_eps)

    return features



def extract_all_features(df):
    """Extract a comprehensive set of movement features from a trajectory.

    Aggregates basic, turning-related, entropy, wavelet, roaming/dwelling,
    frenetic movement, and several enhanced physics/statistical/temporal/
    frequency/behavioral state features.

    Args:
        df: DataFrame with at least `x`, `y`, `speed`, and `turning_angle`.

    Returns:
        dict: Mapping feature_name → value for the provided trajectory.
    """
    features = {}
    
    # Original features
    features.update(calculate_basic_features(df))
    features.update(calculate_turning_features(df))
    features.update(calculate_entropy_features(df))
    features.update(calculate_wavelet_features(df))
    features.update(calculate_roaming_dwelling_score(df))
    features.update(calculate_frenetic_movement(df))

    # Enhanced features
    features.update(calculate_enhanced_physics_features(df))
    features.update(calculate_enhanced_statistical_features(df))
    features.update(calculate_enhanced_temporal_features(df))
    features.update(calculate_enhanced_frequency_features(df))
    features.update(calculate_behavioral_state_features(df))

    # Additional advanced features
    features.update(calculate_additional_features(df))

    # State & behavioral structure features
    features.update(calculate_state_features(df))
    features.update(calculate_burst_features(df))
    features.update(calculate_degradation_features(df))
    features.update(calculate_turn_event_features(df))
    features.update(calculate_terminal_features(df))

    # Final NaN/inf safety clamp (should not trigger anymore)
    for k, v in features.items():
        if not np.isfinite(v):
            features[k] = 0

    return features

def process_all_files(input_dir, output_dir="feature_data"):
    """Process all preprocessed CSVs to compute features and write CSV outputs.

    Args:
        input_dir (str): Base directory containing 'segments' and 'full' subdirectories.
        output_dir (str): Directory to save feature CSV files.

    Returns:
        bool: True if processing completed (files may still be skipped if
        missing), False otherwise.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for data_type in ['full', 'segments']:
        data_dir = os.path.join(input_dir, data_type)
        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} not found. Skipping.")
            continue
        
        output_file = os.path.join(output_dir, f"{data_type}_features.csv")
        
        metadata_path = os.path.join(data_dir, "labels_and_metadata.csv")
        if not os.path.exists(metadata_path):
            print(f"Metadata file not found at {metadata_path}. Skipping.")
            continue
        
        metadata = pd.read_csv(metadata_path)
        print(f"Found {len(metadata)} files in {data_type} metadata")
        
        all_features = []

        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"Processing {data_type}"):
            try:
                if 'relative_path' in metadata.columns:
                    file_path = os.path.join(data_dir, row['relative_path'], row['file'])
                else:
                    file_path = os.path.join(data_dir, row['file'])
                
                if not os.path.exists(file_path):
                    print(f"Warning: File {file_path} not found")
                    continue
                
                # Read and extract features
                df = pd.read_csv(file_path)
                features = extract_all_features(df)
                

                print(f"  Total features extracted: {len(features)}")
                

                
                print("[DEBUG] Feature check complete. Continuing...\n")
                
                # Add metadata information
                features['filename'] = row['file']
                features['label'] = row['label']
                
                # Add additional metadata if available
                if 'death_index' in row:
                    features['death_index'] = row['death_index']
                if 'segment_index' in row:
                    features['segment_index'] = row['segment_index']
                if 'original_file' in row:
                    features['original_file'] = row['original_file']
                
                all_features.append(features)
            except Exception as e:
                print(f"Error processing {row['file']}: {e}")
        
        if all_features:
            features_df = pd.DataFrame(all_features)
            features_df.to_csv(output_file, index=False)
            print(f"Saved features for {len(features_df)} {data_type} data to {output_file}")
        else:
            print(f"Warning: No features extracted for {data_type}")
    
    return True



if __name__ == "__main__":
    input_dir = "preprocessed_data"
    output_dir = "feature_data"
    
    process_all_files(input_dir, output_dir)