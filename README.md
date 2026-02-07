# RAPT-CLIP: Adapter Prompt Temporal CLIP for Video Emotion Recognition

**RAPT-CLIP** is a robust and efficient framework for **Video Emotion Recognition**. Originally designed for the **RAER** dataset, it has been extended to support the **DAISEE** dataset. It leverages the power of **CLIP (Contrastive Language-Image Pre-training)** combined with lightweight adaptation techniques to achieve high performance with minimal computational overhead.

## 🚀 Key Technologies

*   **Backbone:** CLIP (ViT-B/16) - Exploiting strong multimodal (image-text) features.
*   **Prompt Learning (CoOp):** Learnable Context Vectors to optimize text prompts for emotion classes.
*   **Efficient Fine-tuning:** Using **Adapters** to adapt CLIP's visual encoder without retraining the massive backbone.
*   **Temporal Modeling:** **Attention Pooling** mechanism to aggregate frame-level features into video-level representation.
*   **Advanced Loss Functions:**
    *   **Label Distribution Learning (LDL):** Handles label ambiguity in emotion data.
    *   **Decorrelation Loss (DC):** Encourages diverse feature learning.
    *   **Mutual Information Loss (MI):** Maximizes alignment between learnable and hand-crafted prompts.

## 📂 Project Structure

```
├── main.py                 # Entry point for Training and Evaluation
├── train.sh                # Main script to launch Training (Pre-configured)
├── valid.sh                # Script to launch Evaluation
├── trainer.py              # Training loop, validation, and metric logging
├── models/
│   ├── Generate_Model.py   # Main architecture
│   ├── Adapter.py          # Adapter module
│   ├── Prompt_Learner.py   # CoOp Prompt Learning
│   ├── Text.py             # Class definitions (RAER & DAISEE)
│   └── Temporal_Model.py   # Temporal Attention Pooling
├── utils/
│   ├── builders.py         # Model and DataLoader builders
│   ├── loss.py             # Custom Loss functions (LDL, DC, MI)
│   └── utils.py            # Utils (Supports .txt and .csv reading)
└── dataloader/             
    ├── video_dataloader.py # RAER / Generic Loader
    └── video_dataloader_DAISEE.py # DAISEE Specific Loader (CSV support)
```

## 💾 Supported Datasets

### 1. RAER (Rarely Acted Emotion Recognition)
*   **Classes (5):** Neutrality, Enjoyment, Confusion, Fatigue, Distraction.
*   **Format:** Text file annotations (`path/to/video.mp4 num_frames label_id`).

### 2. DAISEE (Dataset for Affective States in E-Environments)
*   **Classes (4):** Engagement, Boredom, Confusion, Frustration.
*   **Format:** CSV file annotations (`VideoPath, Label`).
*   **Data Structure Requirement:**
    ```text
    /path/to/DAISEE/
    ├── DataSet/              # Contains video folders (e.g., Train/, Test/)
    └── Labels/
        ├── TrainLabels.csv
        ├── ValidationLabels.csv
        └── TestLabels.csv
    ```

## 🛠️ Usage

### 1. Training

The project uses `train.sh` as the main configuration file. 

**To Train on DAISEE:**
Ensure `train.sh` is configured with `--dataset DAISEE` and points to your CSV files.

```bash
# Example content of train.sh for DAISEE
python main.py \
  --mode train \
  --dataset DAISEE \
  --root-dir ./DAISEE/DataSet \
  --train-annotation ./DAISEE/Labels/TrainLabels.csv \
  ...
```

Run the script:
```bash
sh train.sh
```

**To Train on RAER:**
Change `--dataset RAER` and update annotation paths in `train.sh`.

### 2. Evaluation

To evaluate a trained model, update `valid.sh` with the path to your checkpoint and the correct dataset configuration.

```bash
sh valid.sh
```

**Key Parameters:**
*   `--dataset`: `RAER` or `DAISEE`.
*   `--text-type`: `prompt_ensemble` (Recommended).
*   `--use-adapter`: `True` (Enable efficient fine-tuning).
*   `--use-ldl`: `True` (Enable Label Distribution Learning).

## 📊 Performance Notes

*   **RAER Target UAR:** ~73% (Validation).
*   **DAISEE:** Optimized with Weighted Sampling (optional) and specialized Prompt Ensembles.
*   **Regularization:** The project uses Mixup, MI Loss, and DC Loss to combat overfitting.

## 🧹 Code Cleanliness

This codebase has been refactored to modularly support multiple datasets via `builders.py` and specialized dataloaders, keeping the core logic clean and extensible.

## 📝 License

[Insert License Here]
