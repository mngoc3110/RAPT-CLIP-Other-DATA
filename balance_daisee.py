import pandas as pd
import numpy as np
import os

# Configuration
input_path = 'DAiSEE/Labels/TrainLabels.csv'
output_path = 'DAiSEE/Labels/TrainLabels_Balanced.csv'
target_column = 'Engagement'
max_samples_per_majority_class = 600
random_seed = 42

def balance_dataset():
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()
    
    if target_column not in df.columns:
        print(f"Error: Column '{target_column}' not found. Available columns: {df.columns.tolist()}")
        return

    print("Original distribution:")
    print(df[target_column].value_counts().sort_index())

    # Separate by class
    df_0 = df[df[target_column] == 0]
    df_1 = df[df[target_column] == 1]
    df_2 = df[df[target_column] == 2]
    df_3 = df[df[target_column] == 3]

    # Downsample majority classes
    if len(df_2) > max_samples_per_majority_class:
        df_2 = df_2.sample(n=max_samples_per_majority_class, random_state=random_seed)
    
    if len(df_3) > max_samples_per_majority_class:
        df_3 = df_3.sample(n=max_samples_per_majority_class, random_state=random_seed)

    # Combine
    balanced_df = pd.concat([df_0, df_1, df_2, df_3])
    
    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    print("\nBalanced distribution:")
    print(balanced_df[target_column].value_counts().sort_index())

    print(f"\nSaving to {output_path}...")
    balanced_df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    balance_dataset()
