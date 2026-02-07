import pandas as pd
import numpy as np
import os

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

    print("Original distribution:")
    print(df[target_column].value_counts().sort_index())

    # Separate by class
    dfs = {}
    for i in range(4):
        dfs[i] = df[df[target_column] == i]

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

    print(f"Saving to {output_path}...")
    balanced_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Train set: Keep aggressive balancing (600) to force learning minority classes
    balance_csv('DAiSEE/Labels/TrainLabels.csv', 'DAiSEE/Labels/TrainLabels_Balanced.csv', 600)
    
    # Validation/Test: Downsample significantly to monitor UAR correctly (prevent class 2 dominance metrics)
    balance_csv('DAiSEE/Labels/ValidationLabels.csv', 'DAiSEE/Labels/ValidationLabels_Balanced.csv', 150)
    balance_csv('DAiSEE/Labels/TestLabels.csv', 'DAiSEE/Labels/TestLabels_Balanced.csv', 150)
    
    print("\nAll balancing tasks completed.")
