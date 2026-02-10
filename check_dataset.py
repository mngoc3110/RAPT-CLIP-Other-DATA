import os
import sys
import torch
from dataloader.video_dataloader_DAISEE import train_data_loader_daisee

# Configuration
ROOT_DIR = "./DAiSEE/DataSet"
TRAIN_CSV = "./DAiSEE/Labels/TrainLabels_Balanced.csv"

def check_load_data():
    print("--- Deep Data Loading Verification ---")
    
    # 1. Init Dataset
    # Lưu ý: use_face=True để kiểm tra cả Haar Cascade
    try:
        dataset = train_data_loader_daisee(
            root_dir=os.path.join(ROOT_DIR, 'Train'),
            csv_file=TRAIN_CSV,
            num_segments=16, 
            duration=1,
            image_size=224,
            use_face=True
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize dataset: {e}")
        return
    
    if len(dataset) == 0:
        print("Error: No samples found in CSV.")
        return

    # 2. Inspect First Sample Path
    video_path, label = dataset.samples[0]
    print(f"\n[Sample 0] Path: {video_path}")
    
    video_dir = os.path.dirname(video_path)
    frames_dir = os.path.join(video_dir, "frames")
    
    if os.path.exists(frames_dir) and os.path.isdir(frames_dir):
        print(f"[SUCCESS] Found 'frames' folder: {frames_dir}")
        num_imgs = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg') or f.endswith('.png')])
        print(f"         Contains {num_imgs} images.")
    else:
        print(f"[WARNING] 'frames' folder NOT found. Will fallback to AVI video reading.")

    # 3. Actually Load Data (Run __getitem__)
    print("\nAttempting to load data tensors (Running __getitem__)... This might take a few seconds due to face detection...")
    try:
        # face_t, full_t, label
        face_tensor, body_tensor, label_out = dataset[0]
        
        print(f"\n[SUCCESS] Data Loaded Successfully!")
        print(f"  - Label: {label_out}")
        print(f"  - Face Tensor Shape: {face_tensor.shape} (Segments, Channel, H, W)")
        print(f"  - Body Tensor Shape: {body_tensor.shape} (Segments, Channel, H, W)")
        
        # Kiểm tra xem có phải là tensor rỗng (dummy) không
        if torch.all(face_tensor == 0):
            print("  - [WARNING] Face tensor is all zeros. This might be a dummy due to missing file.")
        else:
            print("  - [OK] Tensors contain data.")

    except Exception as e:
        print(f"\n[ERROR] Failed to load data during __getitem__: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_load_data()