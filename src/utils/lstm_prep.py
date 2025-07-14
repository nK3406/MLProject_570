import dask.dataframe as dd
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from dask.diagnostics import ProgressBar
import os

# Step 1: Load data
def load_data(parquet_path, meta_path):
    # Load parquet with dask
    ddf = dd.read_parquet(parquet_path, blocksize="100MB")  # Adjust blocksize based on memory
    meta = pd.read_csv(meta_path)
    return ddf, meta

# Step 2: Custom linear interpolation for dask
def custom_interpolate(df):
    # Convert to pandas for interpolation within each partition
    df = df.interpolate(method='linear', limit_direction='both')
    return df

def impute_missing(ddf):
    # Apply custom interpolation to each partition
    ddf = ddf.map_partitions(custom_interpolate)
    # Drop sensors with >50% missing values
    missing_ratio = ddf.isna().mean().compute()
    valid_columns = missing_ratio[missing_ratio < 0.5].index
    ddf = ddf[valid_columns]
    return ddf

# Step 3: Preprocess timestamps
def preprocess_timestamps(ddf):
    # Convert timestamps to datetime
    ddf['timestamp'] = dd.to_datetime(ddf['timestamp'], unit='ns')
    ddf = ddf.set_index('timestamp')
    # Resample to 5-minute intervals
    ddf = ddf.resample('5T').mean()  # Resampling in dask, interpolation handled separately
    ddf = ddf.map_partitions(custom_interpolate)  # Apply interpolation after resampling
    return ddf

# Step 4: Merge metadata
def merge_metadata(ddf, meta):
    sensor_ids = [col for col in ddf.columns if col != 'timestamp']
    meta = meta[meta['ID'].astype(str).isin(sensor_ids)]
    
    # Encode categorical features
    categorical_cols = ['County', 'Fwy', 'Direction']
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(meta[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols), index=meta['ID'])
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_cols = ['Lat', 'Lng', 'Lanes']
    meta[numerical_cols] = scaler.fit_transform(meta[numerical_cols])
    
    # Combine metadata
    meta_features = pd.concat([meta[['ID'] + numerical_cols], encoded_df], axis=1)
    return meta_features

# Step 5: Create sequences with overlap
def create_sequences(ddf, meta_features, seq_length=12, overlap=12):
    sequences = []
    targets = []
    sensor_ids = [col for col in ddf.columns if col != 'timestamp']
    
    # Normalize traffic density
    scaler = MinMaxScaler()
    ddf_np = ddf.compute().values  # Compute to numpy for sequence creation
    ddf_np = scaler.fit_transform(ddf_np)
    
    for sensor_idx, sensor in enumerate(sensor_ids):
        sensor_data = ddf_np[:, sensor_idx]
        sensor_meta = meta_features[meta_features['ID'] == int(sensor)].drop('ID', axis=1).values
        
        for i in range(0, len(sensor_data) - seq_length, seq_length - overlap):
            seq = sensor_data[i:i + seq_length]
            if len(seq) == seq_length:  # Ensure full sequence
                target = sensor_data[i + seq_length]
                seq_meta = np.tile(sensor_meta, (seq_length, 1))
                seq_combined = np.concatenate([seq.reshape(-1, 1), seq_meta], axis=1)
                sequences.append(seq_combined)
                targets.append(target)
    
    return np.array(sequences), np.array(targets)

# Step 6: Split data
def split_data(sequences, targets, train_ratio=0.7, val_ratio=0.2):
    n = len(sequences)
    train_idx = int(train_ratio * n)
    val_idx = int((train_ratio + val_ratio) * n)
    
    train_seq = sequences[:train_idx]
    val_seq = sequences[train_idx:val_idx]
    test_seq = sequences[val_idx:]
    
    train_targets = targets[:train_idx]
    val_targets = targets[train_idx:val_idx]
    test_targets = targets[val_idx:]
    
    return train_seq, val_seq, test_seq, train_targets, val_targets, test_targets

# Step 7: Save sequences incrementally
def save_sequences(sequences, targets, prefix, output_dir="preprocessed_data"):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{prefix}_sequences.npy"), sequences)
    np.save(os.path.join(output_dir, f"{prefix}_targets.npy"), targets)

# Main preprocessing pipeline
def preprocess_pipeline(parquet_path, meta_path, seq_length=12, overlap=12):
    # Load data
    ddf, meta = load_data(parquet_path, meta_path)
    
    # Preprocess
    with ProgressBar():
        ddf = impute_missing(ddf)
        ddf = preprocess_timestamps(ddf)
        meta_features = merge_metadata(ddf, meta)
        
        # Create sequences
        sequences, targets = create_sequences(ddf, meta_features, seq_length, overlap)
    
    # Split data
    train_seq, val_seq, test_seq, train_targets, val_targets, test_targets = split_data(sequences, targets)
    
    # Save preprocessed data
    save_sequences(train_seq, train_targets, "train")
    save_sequences(val_seq, val_targets, "val")
    save_sequences(test_seq, test_targets, "test")
    meta_features.to_csv(os.path.join("preprocessed_data", "processed_meta.csv"), index=False)
    
    return train_seq, val_seq, test_seq, train_targets, val_targets, test_targets

# Example usage
if __name__ == "__main__":
    parquet_path = "data/processed/largest_wide_2017"
    meta_path = "/home/bistek/Downloads/archive_largest/ca_meta.csv"
    train_seq, val_seq, test_seq, train_targets, val_targets, test_targets = preprocess_pipeline(
        parquet_path, meta_path, seq_length=12, overlap=12
    )
    print(f"Train shape: {train_seq.shape}, Val shape: {val_seq.shape}, Test shape: {test_seq.shape}")
