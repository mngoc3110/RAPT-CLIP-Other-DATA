import os
import csv
import random
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils import data
import torchvision
from collections import Counter
from dataloader.video_transform import (
    GroupRandomHorizontalFlip, GroupRandomCrop, GroupNormalize, 
    GroupScale, GroupCenterCrop, ToTorchFormatTensor, Stack, GroupResize
)

# =========================
# 1) Label mapping (DAiSEE)
# Engagement Levels: 0, 1, 2, 3
# =========================
DAISEE_LABEL_MAP = {
    "very low": 0,
    "low": 1,
    "high": 2,
    "very high": 3
}
REVERSE_DAISEE_LABEL_MAP = {v: k for k, v in DAISEE_LABEL_MAP.items()}

# =========================
# 2) Video Transforms (Consistent across frames)
# =========================
def default_train_transform(image_size=224):
    return torchvision.transforms.Compose([
        GroupResize(int(image_size * 1.2)), # Resize first to ensure all frames are same size
        GroupRandomHorizontalFlip(),
        GroupRandomCrop(image_size),
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

def default_test_transform(image_size=224):
    return torchvision.transforms.Compose([
        GroupResize(int(image_size * 1.2)), # Ensure test images are also consistently sized
        GroupCenterCrop(image_size),
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

# =========================
# 3) TSN sampling
# =========================
def get_train_indices(num_frames, num_segments, duration):
    average_duration = (num_frames - duration + 1) // num_segments
    if average_duration > 0:
        offsets = np.multiply(list(range(num_segments)), average_duration) + \
                  np.random.randint(average_duration, size=num_segments)
    elif num_frames > num_segments:
        offsets = np.sort(np.random.randint(num_frames - duration + 1, size=num_segments))
    else:
        offsets = np.pad(np.array(list(range(num_frames))), (0, num_segments - num_frames), "edge")
    return offsets

def get_test_indices(num_frames, num_segments, duration):
    if num_frames > num_segments + duration - 1:
        tick = (num_frames - duration + 1) / float(num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
    else:
        offsets = np.pad(np.array(list(range(num_frames))), (0, num_segments - num_frames), "edge")
    return offsets

# =========================
# 4) (Optional) Face detector
# =========================
class HaarFaceCropper:
    def __init__(self):
        self.haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = None

    def crop_face(self, bgr_img, margin=20):
        if self.detector is None:
            self.detector = cv2.CascadeClassifier(self.haar_path)
            
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return bgr_img

        # lấy face lớn nhất
        x, y, w, h = sorted(faces, key=lambda t: t[2]*t[3], reverse=True)[0]
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(bgr_img.shape[1], x + w + margin)
        y2 = min(bgr_img.shape[0], y + h + margin)
        return bgr_img[y1:y2, x1:x2]

# =========================
# 5) DAiSEE Dataset
# =========================
class DaiseeVideoDataset(data.Dataset):
    def __init__(
        self,
        csv_file,
        root_dir="",
        mode="train",
        num_segments=16,
        duration=1,
        image_size=224,
        transform=None,
        use_face=True,
        face_cropper=None
    ):
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.mode = mode
        self.num_segments = num_segments
        self.duration = duration
        self.image_size = image_size

        self.transform = transform if transform is not None else (
            default_train_transform(image_size) if mode == "train" else default_test_transform(image_size)
        )

        self.use_face = use_face
        self.face_cropper = face_cropper if face_cropper is not None else HaarFaceCropper()

        self.samples = self._read_csv(csv_file)
        print(f"[DAiSEE] Loaded {len(self.samples)} videos from {csv_file}")
        
        labels = [s[1] for s in self.samples]
        counts = Counter(labels)
        print(f"[DAiSEE] Label distribution for {mode}: {dict(counts)}")

    def _read_csv(self, csv_file):
        items = []
        if not os.path.exists(csv_file):
            print(f"[Error] CSV not found: {csv_file}")
            return items

        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        if not rows:
            return items

        # Headers check
        header = rows[0]
        label_col_idx = 2  # Default to 2
        
        # Try to find "Engagement" column dynamically
        found_header = False
        target_label = "Engagement"
        
        # Check if first row looks like a header
        # Usually headers contain "Clip" or the target label
        if any("Clip" in h for h in header) or any(target_label.lower() in h.lower() for h in header):
            found_header = True
            try:
                # Find column that contains "Engagement" (case insensitive)
                label_col_idx = next(i for i, h in enumerate(header) if target_label.lower() in h.lower())
                print(f"[DAiSEE] Found '{target_label}' at column index {label_col_idx} in {os.path.basename(csv_file)}")
            except StopIteration:
                print(f"[DAiSEE] Warning: Header found but '{target_label}' column not found. Defaulting to index {label_col_idx}.")
        else:
            print(f"[DAiSEE] Warning: No header detected (first row: {header}). Defaulting to index {label_col_idx}.")

        start_idx = 1 if found_header else 0
        
        for i in range(start_idx, len(rows)):
            self._process_row(rows[i], items, label_col_idx)

        # Debug: Print first 5 samples
        print(f"[DAiSEE] First 5 loaded samples from {os.path.basename(csv_file)}:")
        for k in range(min(5, len(items))):
            print(f"  {items[k]}")
            
        return items

    def _process_row(self, row, items, label_col_idx=2):
        if len(row) <= label_col_idx: return
        rel_path = row[0].strip()
        if not rel_path.lower().endswith(('.avi', '.mp4', '.mov')):
             rel_path += ".avi"
             
        file_name = os.path.basename(rel_path)
        base_name = os.path.splitext(file_name)[0]
        if len(base_name) >= 6 and base_name[:6].isdigit():
            subject_id = base_name[:6]
            nested_rel_path = os.path.join(subject_id, base_name, file_name)
        else:
            nested_rel_path = rel_path

        label_str = row[label_col_idx].strip()
        if not label_str.isdigit(): return
        label_id = int(label_str)
        
        if 0 <= label_id <= 3:
            video_path = os.path.join(self.root_dir, nested_rel_path) if self.root_dir else nested_rel_path
            items.append((video_path, label_id))

    def _get_num_frames(self, cap):
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return max(n, 0)

    def _read_frame_at(self, cap, frame_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None: return None
        return frame

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_path, label = self.samples[index]
        
        # Determine if we are using frames or video file
        # Check for "frames" folder in the same directory as the video or replacing the video file
        # video_path e.g.: .../1100042011/1100042011.avi
        # frames path expected: .../1100042011/frames/
        
        video_dir = os.path.dirname(video_path)
        frames_dir = os.path.join(video_dir, "frames")
        
        use_frames = False
        if os.path.exists(frames_dir) and os.path.isdir(frames_dir):
             use_frames = True
             # Check if populated
             all_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg') or f.endswith('.png')])
             num_frames = len(all_files)
             if num_frames <= 0:
                 use_frames = False # Fallback to AVI if empty
        
        if not use_frames:
            # Fallback to existing AVI logic
            if not os.path.exists(video_path):
                print(f"[Warning] File not found: {video_path}")
                dummy = torch.zeros((self.num_segments * self.duration, 3, self.image_size, self.image_size))
                return dummy, dummy, label

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[Warning] Could not open video: {video_path}")
                dummy = torch.zeros((self.num_segments * self.duration, 3, self.image_size, self.image_size))
                return dummy, dummy, label

            num_frames = self._get_num_frames(cap)
        
        if num_frames <= 0:
            if not use_frames: cap.release()
            dummy = torch.zeros((self.num_segments * self.duration, 3, self.image_size, self.image_size))
            return dummy, dummy, label

        if self.mode == "train":
            indices = get_train_indices(num_frames, self.num_segments, self.duration)
        else:
            indices = get_test_indices(num_frames, self.num_segments, self.duration)

        indices = np.clip(indices, 0, num_frames - 1)
        full_imgs, face_imgs = [], []

        for seg_ind in indices:
            p = int(seg_ind)
            for _ in range(self.duration):
                if use_frames:
                    # Read from image file (1-based indexing for filenames usually)
                    # File pattern checked earlier: 1100042011_00001.jpg
                    # We need to robustly find the file corresponding to index p
                    # Assuming sorted list matches indices 0..N-1
                    frame_filename = all_files[p]
                    frame_path = os.path.join(frames_dir, frame_filename)
                    frame = cv2.imread(frame_path)
                else:
                    frame = self._read_frame_at(cap, p)
                
                if frame is None:
                    # Use size from previous frame if available, otherwise default
                    if len(full_imgs) > 0:
                        w, h = full_imgs[0].size
                        frame = np.zeros((h, w, 3), dtype=np.uint8)
                    else:
                        frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

                # --- Context Stream (Body/Background) ---
                # Use Full Frame as Body/Context
                full_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                full_pil = Image.fromarray(full_rgb)

                # --- Face Stream ---
                if self.use_face:
                    # Crop face from the full frame
                    face_bgr = self.face_cropper.crop_face(frame, margin=20)
                    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                else:
                    face_pil = full_pil

                full_imgs.append(full_pil)
                face_imgs.append(face_pil)
                
                # Increment frame pointer safely
                p = min(p + 1, num_frames - 1)

        if not use_frames:
            cap.release()
            
        # Apply transforms to the group of images (Temporal Consistency)
        # transform expects a list of PIL Images
        full_t = self.transform(full_imgs) # returns (C*T, H, W)
        face_t = self.transform(face_imgs) # returns (C*T, H, W)

        # Reshape to (T, C, H, W) as expected by the model
        # The ToTorchFormatTensor returns (C, H, W) but Stack concatenates channels.
        # Wait, let's check VideoTransform.
        # Stack(roll=False) -> concatenates in dim 2 (channels). 
        # So we get H x W x (C*T).
        # ToTorchFormatTensor -> (C*T, H, W).
        # We need (T, C, H, W).
        
        c = 3
        t = self.num_segments * self.duration
        
        full_t = full_t.view(t, c, self.image_size, self.image_size)
        face_t = face_t.view(t, c, self.image_size, self.image_size)

        return face_t, full_t, label

def train_data_loader_daisee(root_dir, csv_file, num_segments, duration, image_size, use_face=True):
    return DaiseeVideoDataset(
        csv_file=csv_file, root_dir=root_dir, mode="train",
        num_segments=num_segments, duration=duration, image_size=image_size,
        transform=default_train_transform(image_size), use_face=use_face
    )

def test_data_loader_daisee(root_dir, csv_file, num_segments, duration, image_size, use_face=True):
    return DaiseeVideoDataset(
        csv_file=csv_file, root_dir=root_dir, mode="test",
        num_segments=num_segments, duration=duration, image_size=image_size,
        transform=default_test_transform(image_size), use_face=use_face
    )