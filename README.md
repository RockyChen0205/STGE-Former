# STGE-Former: Spatial-Temporal Graph-Enhanced Transformer for EEG-Based Major Depressive Disorder Detection

![Framework Diagram](assets/framework.svg)

## Overview

This repository contains the official implementation of **STGE-Former** (Spatial-Temporal Graph-Enhanced Transformer), a novel deep learning framework for detecting Major Depressive Disorder (MDD) from EEG signals. Our model achieves state-of-the-art performance on the MODMA dataset by effectively capturing both spatial and temporal dependencies in EEG data.


## Model Architecture

The overall framework of the STGE-Former mainly comprises four components:

1. **Input Pipeline**: Preprocesses and prepares EEG signals for feature extraction
2. **Spatial Attention Stream**: Models functional connectivity among different brain regions using graph attention mechanisms
3. **Temporal Graph-Enhanced Attention Stream**: Captures brain activity correlations across various time intervals with enhanced transformer architecture
4. **Classification Head**: Integrates spatiotemporal features for final depression detection

As shown in the framework diagram, these components work collaboratively to extract and fuse both spatial and temporal features for optimal depression classification performance.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+ (for GPU acceleration)

### Setup
```bash
git clone [repository-url]
cd pycharm_project_29
pip install -r requirements.txt
```

## Dataset Preparation

### MODMA Dataset
1. Download the MODMA dataset from the official source
2. Place the raw EEG files in the `data/` directory
3. Run preprocessing scripts:

```bash
cd preprocess
python eeg_slice.py
```

### Preprocessed Data
The preprocessing pipeline generates:
- Time-series segments in `.set` format
- Corresponding label files
- Processed numpy arrays in `after_process_data/`

## Usage

### Training
Run 10-fold cross-validation on MODMA dataset:

```bash
# For STGE-Former
bash STGEFormer_MODMA_10fold.sh

# Alternative models
bash SFormer_MODMA_10fold.sh
bash STFormer_MODMA_10fold.sh
bash TGEFormer_MODMA_10fold.sh
```

### Evaluation
Metrics are automatically computed and saved to `results/metrics.txt`
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

## Project Structure

```
pycharm_project_29/
├── data/                 # Raw EEG data files
├── after_process_data/   # Processed numpy arrays
├── data_provider/        # Data loading utilities
├── model/               # Model architectures
│   ├── STGEFormer.py    # Main model implementation
│   ├── Encoder.py       # Transformer encoder
│   ├── embedding.py     # Feature embedding layers
│   └── self_attention.py # Attention mechanisms
├── preprocess/          # Data preprocessing scripts
├── utils/               # Utility functions
├── results/             # Training results and metrics
└── assets/             # Framework diagrams
```

