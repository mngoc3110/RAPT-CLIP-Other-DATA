import pandas as pd
import os

files = [
    'DAiSEE/Labels/TrainLabels.csv',
    'DAiSEE/Labels/ValidationLabels.csv',
    'DAiSEE/Labels/TestLabels.csv'
]

print("-" * 60)
print("CHECKING LABEL FILES CONTENT")
print("-" * 60)

for f in files:
    if not os.path.exists(f):
        print(f"File NOT FOUND: {f}")
        continue
        
    try:
        # Read CSV
        df = pd.read_csv(f)
        
        # Clean column names (strip spaces)
        df.columns = df.columns.str.strip()
        
        # Identify Target Column
        target = 'Engagement'
        if target not in df.columns:
            # Fallback search
            candidates = [c for c in df.columns if 'ngagement' in c]
            if candidates:
                target = candidates[0]
            else:
                print(f"File: {os.path.basename(f)}")
                print(f"  STATUS: ERROR - Column 'Engagement' not found.")
                print(f"  Columns detected: {list(df.columns)}")
                print("-" * 30)
                continue
        
        # Check Values
        unique_vals = sorted(df[target].unique())
        min_val = min(unique_vals)
        max_val = max(unique_vals)
        
        print(f"File: {os.path.basename(f)}")
        print(f"  Target Column: '{target}'")
        print(f"  Unique Values: {unique_vals}")
        
        # Diagnosis
        if min_val >= 0 and max_val <= 3:
            print(f"  STATUS: OK (Standard 0-3 range)")
        elif min_val >= 1 and max_val <= 4:
            print(f"  STATUS: WARNING (Likely 1-4 range). Needs shifting.")
        else:
            print(f"  STATUS: UNKNOWN/MIXED (Range {min_val}-{max_val})")
            
    except Exception as e:
        print(f"File: {os.path.basename(f)}")
        print(f"  STATUS: CRASHED reading file. Error: {e}")
    
    print("-" * 30)
