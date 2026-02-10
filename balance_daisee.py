import pandas as pd
import numpy as np
import os
import argparse

# Configuration
target_column = 'Engagement'
random_seed = 42

def balance_csv(input_path, output_path, max_samples_per_majority_class):
    if not os.path.exists(input_path):
        print(f"Skipping {input_path}: File not found.")
        return

    print(f"\nProcessing {input_path}...")
    df = pd.read_csv(input_path)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    if target_column not in df.columns:
        print(f"Error: Column '{target_column}' not found in {input_path}.")
        return

    # Check for 1-based indexing (1-4) and convert to 0-3 if necessary
    unique_labels = sorted(df[target_column].unique())
    print(f"Unique labels found: {unique_labels}")
    
    if any(l > 3 for l in unique_labels):
        print("Detected labels > 3. Assuming 1-based indexing (1-4). Shifting to 0-3...")
        df[target_column] = df[target_column] - 1
        unique_labels = sorted(df[target_column].unique())
        print(f"New unique labels: {unique_labels}")
        
    if not all(0 <= l <= 3 for l in unique_labels):
         print(f"Warning: Labels out of expected range 0-3 after adjustment: {unique_labels}")

    print("Original distribution:")
    print(df[target_column].value_counts().sort_index())

    # Separate by class
    dfs = {}
    for i in range(4):
        # Handle missing classes gracefully
        if i in unique_labels:
            dfs[i] = df[df[target_column] == i]
        else:
            dfs[i] = pd.DataFrame(columns=df.columns)

    # Downsample majority classes
    frames = []
    for i in range(4):
        subset = dfs[i]
        if len(subset) > max_samples_per_majority_class:
            subset = subset.sample(n=max_samples_per_majority_class, random_state=random_seed)
        frames.append(subset)

    # Combine
    balanced_df = pd.concat(frames)
    
    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    print("Balanced distribution:")
    print(balanced_df[target_column].value_counts().sort_index())

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving to {output_path}...")
    balanced_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balance DAiSEE Dataset Labels")
    
    # Defaults set to local structure
    parser.add_argument('--train-in', type=str, default='DAiSEE/Labels/TrainLabels.csv', help='Path to input TrainLabels.csv')
    parser.add_argument('--val-in', type=str, default='DAiSEE/Labels/ValidationLabels.csv', help='Path to input ValidationLabels.csv')
    parser.add_argument('--test-in', type=str, default='DAiSEE/Labels/TestLabels.csv', help='Path to input TestLabels.csv')
    
    parser.add_argument('--train-out', type=str, default='DAiSEE/Labels/TrainLabels_Balanced.csv', help='Path to output balanced Train CSV')
    parser.add_argument('--val-out', type=str, default='DAiSEE/Labels/ValidationLabels_Balanced.csv', help='Path to output balanced Validation CSV')
    parser.add_argument('--test-out', type=str, default='DAiSEE/Labels/TestLabels_Balanced.csv', help='Path to output balanced Test CSV')
    
    args = parser.parse_args()

    # Train set: Keep aggressive balancing (600) to force learning minority classes
    balance_csv(args.train_in, args.train_out, 600)
    
    # Validation/Test: Downsample significantly to monitor UAR correctly
    balance_csv(args.val_in, args.val_out, 150)
    balance_csv(args.test_in, args.test_out, 150)
    
    print("\nAll balancing tasks completed.")
