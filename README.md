# RAPT-CLIP: RAER Adapter Prompt Temporal CLIP

**RAPT-CLIP** is a robust and efficient framework for **Video Emotion Recognition**, specifically designed for the **RAER** dataset. It leverages the power of **CLIP (Contrastive Language-Image Pre-training)** combined with lightweight adaptation techniques to achieve high performance with minimal computational overhead.

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
├── main.py                 # Entry point for Training and Evaluation (Coordinator)
├── train.sh                # Script to launch Training (Configured with "Safe Mode")
├── valid.sh                # Script to launch Evaluation
├── trainer.py              # Training loop, validation, and metric logging
├── models/
│   ├── Generate_Model.py   # Main architecture definition (Forward pass)
│   ├── Adapter.py          # Lightweight Adapter module
│   ├── Prompt_Learner.py   # CoOp Prompt Learning module
│   └── Temporal_Model.py   # Temporal Attention Pooling module
├── utils/
│   ├── builders.py         # Model and DataLoader construction
│   ├── loss.py             # Custom Loss functions (LDL, DC, MI)
│   └── utils.py            # Helper functions
└── dataloader/             # Video data loading and preprocessing
```

## 🛠️ Usage

### 1. Training
The training configuration is optimized in `train.sh` (Safe Mode: Epochs 60, Accumulation 4, Mixup 0.2).

```bash
sh train.sh
```

**Key Parameters in `train.sh`:**
*   `--use-adapter`: Enable Adapter for efficient tuning.
*   `--use-ldl`: Enable Label Distribution Learning.
*   `--lambda_mi / --lambda_dc`: Regularization strengths (0.1).
*   `--exper-name`: Name for the output folder (check `outputs/`).

### 2. Evaluation
To evaluate a trained model, update the `--eval-checkpoint` path in `valid.sh` to point to your `model_best.pth`.

```bash
sh valid.sh
```

## 📊 Performance Notes

*   **Target UAR:** ~73% (Validation).
*   **Regularization:** The project uses Mixup, MI Loss, and DC Loss to combat overfitting, which is common in small/medium video datasets like RAER.

## 🧹 Code Cleanliness

This codebase has been rigorously refactored to remove redundant experimental features (MoCo, Slerp, Focal Loss) and focuses purely on the most effective configuration (**Attention Pool + LDL + Adapter**).

## 📝 License

[Insert License Here]