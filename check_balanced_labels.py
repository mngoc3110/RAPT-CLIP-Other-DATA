import pandas as pd
import os

files = [
    'DAiSEE/Labels/TrainLabels_Balanced.csv',
    'DAiSEE/Labels/ValidationLabels_Balanced.csv',
    'DAiSEE/Labels/TestLabels_Balanced.csv'
]

print("-" * 60)
print("CHECKING BALANCED LABEL FILES")
print("-" * 60)

for f in files:
    if not os.path.exists(f):
        print(f"File NOT FOUND: {f}")
        continue
        
    try:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        target = 'Engagement'
        
        if target in df.columns:
            unique_vals = sorted(df[target].unique())
            counts = df[target].value_counts().sort_index().to_dict()
            
            print(f"File: {os.path.basename(f)}")
            print(f"  Total Samples: {len(df)}")
            print(f"  Unique Labels: {unique_vals}")
            print(f"  Distribution:  {counts}")
            
            if all(v <= 3 for v in unique_vals):
                print(f"  STATUS: OK")
            else:
                print(f"  STATUS: ERROR - Labels found outside 0-3 range!")
        else:
            print(f"File: {os.path.basename(f)}")
            print(f"  STATUS: ERROR - Column 'Engagement' not found.")
            
    except Exception as e:
        print(f"File: {os.path.basename(f)}")
        print(f"  STATUS: CRASHED. Error: {e}")
    
    print("-" * 30)
