# 🧠 NeuroSeg-ICH

> **Neural segmentation pipeline for automated Intracranial Hemorrhage (ICH) detection from brain CT scans using Residual U-Net**

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square)
![Best Dice](https://img.shields.io/badge/Best%20Dice-0.6304-brightgreen?style=flat-square)
![Task](https://img.shields.io/badge/Task-Medical%20Segmentation-purple?style=flat-square)

---

## 📌 Overview

**NeuroSeg-ICH** is a deep learning pipeline for detecting and segmenting Intracranial Hemorrhage (ICH) — bleeding inside the brain — from CT scan slices. Early and accurate detection of ICH is critical in clinical settings, as delayed diagnosis can lead to severe neurological damage or death.

This project implements a custom **Residual U-Net (ResUNet)** architecture written from scratch, trained on real clinical CT data from PhysioNet. The best model was saved at **epoch 15** with a validation Dice of **0.6304**, and the inference pipeline uses adaptive threshold optimization to maximize segmentation quality.

---

## 🏆 Results

| Metric | Value |
|---|---|
| **Best Validation Dice** | **0.6304** (saved at Epoch 15) |
| Final Validation Dice (Threshold=0.1) | 0.5860 |
| Best Single Slice Dice | 0.940 |
| Validation IoU | ~0.40 |
| Validation Precision | ~0.74 |
| Validation Recall | ~0.44 |
| Best Threshold | 0.1 (adaptive search 0.1–0.9) |
| Training Epochs | 50 |

### Threshold Search Output
```
Threshold 0.10 -> Dice: 0.5860  ← Best
Threshold 0.15 -> Dice: 0.5823
Threshold 0.20 -> Dice: 0.5767
...
Threshold 0.90 -> Dice: 0.4934

Best Threshold: 0.1
Final Validation Dice (Best Threshold): 0.5860
Loaded model from epoch 15 with Dice: 0.6304
```

---

## 🖼️ Visual Results

### Best Inference Slice — Dice Score: 0.940
![Final Submission](final_submission_figure.png)

### Training Curves — Loss & Metrics over 50 Epochs
![Training Curves](training_curves.png)

### Clinical Evaluation Map
![Clinical Evaluation](clinical_evaluation_map.png)

### Medical Overlay Visualization
![Medical Overlay](final_medical_overlay.png)

---

## 🎯 Key Features

- ✅ Custom **ResUNet architecture** with true residual connections — written from scratch
- ✅ Combined **BCE + Dice Loss** to handle extreme class imbalance (pos_weight=20)
- ✅ **Adaptive threshold search** (0.1 → 0.9) at inference for best Dice
- ✅ **ReduceLROnPlateau** scheduler for dynamic learning rate adjustment
- ✅ Comprehensive evaluation — **Dice, IoU, Precision, Recall** tracked per epoch
- ✅ Rich visualizations — CT overlays, ground truth contours, prediction contours, error maps

---

## 🏗️ Architecture — ResUNet

The model is a **Residual U-Net** — a U-Net backbone enhanced with residual skip connections for better gradient flow and feature reuse.

```
Input [B, 3, H, W]
    │
    ▼
┌─────────────────────────────────────────────┐
│  ENCODER                                    │
│  StemBlock      → [B,  64, H,    W   ]      │
│  EncoderStage2  → [B, 128, H/2,  W/2 ]      │
│  EncoderStage3  → [B, 256, H/4,  W/4 ]      │
│  EncoderStage4  → [B, 512, H/8,  W/8 ]      │
└─────────────────────────────────────────────┘
    │  (skip connections saved at each stage)
    ▼
┌─────────────────────────────────────────────┐
│  BOTTLENECK  → [B, 1024, H/16, W/16]        │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  DECODER                                    │
│  DecoderStage4  → [B, 512, H/8,  W/8 ]      │
│  DecoderStage3  → [B, 256, H/4,  W/4 ]      │
│  DecoderStage2  → [B, 128, H/2,  W/2 ]      │
│  DecoderStage1  → [B,  64, H,    W   ]      │
└─────────────────────────────────────────────┘
    │
    ▼
Output Logits [B, 1, H, W]
```

Each **ResidualBlock** applies:
```
x ──► BNReLUConv ──► BNReLUConv ──► (+) ──► out
│                                     ▲
└────────── shortcut (1×1 conv) ──────┘
```

---

## 📦 Dataset

Download the **CT-ICH dataset** from PhysioNet:

👉 [https://physionet.org/content/ct-ich/1.3.1/](https://physionet.org/content/ct-ich/1.3.1/)

After downloading, extract and organize as:

```
DataV1/
└── CV0/
    ├── train/
    │   ├── image/
    │   └── label/
    └── validate/
        ├── image/
        └── label/
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/hrrssshhhhhh/NeuroSeg-ICH.git
cd NeuroSeg-ICH
```

### 2. Install Dependencies
```bash
pip install torch torchvision matplotlib numpy
```

### 3. Verify Data Loading
```bash
python test_loader.py
```

### 4. Train the Model
```bash
python train.py
```

### 5. Run Inference & Visualize
```bash
python inference.py
```

---

## 📁 Project Structure

```
NeuroSeg-ICH/
├── Models/
│   └── Models/
│       └── Resunet/
│           ├── model.py          # ResUNet architecture (written from scratch)
│           ├── dataset.py        # ICHDataset data loader
│           └── prepare_data.py   # Data preprocessing
├── train.py                      # Training loop with metrics logging
├── inference.py                  # Adaptive threshold search + visualization
├── test_loader.py                # Data pipeline validation
├── final_submission_figure.png   # Best slice visualization (Dice: 0.940)
├── training_curves.png           # Loss & metric curves over 50 epochs
├── clinical_evaluation_map.png   # Clinical evaluation visualization
├── final_medical_overlay.png     # Medical overlay visualization
├── README.md
└── LICENSE
```

---

## ⚙️ Training Details

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 5e-5 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Loss Function | BCE (pos_weight=20) + Dice Loss |
| Batch Size | 4 |
| Epochs | 50 |
| Best Epoch | 15 |
| Input Channels | 3 (RGB CT slices) |
| Output Classes | 1 (binary segmentation) |

### Why BCE + Dice Loss?
ICH is a highly **imbalanced** segmentation task — hemorrhage pixels are a tiny fraction of the total CT slice. BCE with `pos_weight=20` forces the model to pay more attention to the rare positive class, while Dice Loss directly optimizes the overlap metric.

---

## 📊 Metrics Tracked Per Epoch

- **Dice Score** — primary metric, measures segmentation overlap
- **IoU (Jaccard Index)** — intersection over union
- **Precision** — of all predicted hemorrhage pixels, how many are correct (~0.74)
- **Recall** — of all actual hemorrhage pixels, how many are detected (~0.44)

---

## 🔬 Clinical Relevance

Intracranial Hemorrhage is a **medical emergency** requiring rapid diagnosis. Manual CT interpretation by radiologists is time-consuming and subject to variability. Automated segmentation tools like NeuroSeg-ICH can:

- Assist radiologists in rapid triage
- Provide quantitative hemorrhage volume estimates
- Reduce diagnostic errors in high-workload environments

---

## 👤 Author

**Harsh Jogani**
B.Tech Final Year Project

- GitHub: [@hrrssshhhhhh](https://github.com/hrrssshhhhhh)

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset: [PhysioNet CT-ICH](https://physionet.org/content/ct-ich/1.3.1/) by Hssayeni et al.
- Architecture inspired by: [Road Extraction by Deep Residual U-Net](https://arxiv.org/abs/1711.10684) (Zhang et al., 2018)

---

*If you find this project useful, please ⭐ star the repository!*

