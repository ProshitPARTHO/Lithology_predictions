# README

## Project Overview

This project implements a machine learning pipeline for **lithology classification** from well-log data. The goal is to predict geological rock types (lithofacies) based on various well-logging measurements and spatial data.

The pipeline addresses the common challenge of **class imbalance** in geological datasets by combining:
1. **CTGAN (Conditional Tabular GAN)** for synthetic data augmentation of minority classes
2. **Deep Neural Networks** with attention mechanisms for classification

## Dataset
the dataset can be found at: https://zenodo.org/records/4351156/files/LAS_files_Force_2020_all_wells_train_test_blind_hidden_final.zip?download=1
The dataset contains well-logging data with the following key features:

### Spatial & Identifier Columns
- `WELL`: Well identifier
- `DEPTH_MD`: Depth in meters
- `X_LOC`, `Y_LOC`, `Z_LOC`: Spatial coordinates
- `GROUP`, `FORMATION`: Geological group/formation names

### Well-Log Measurements (Numerical)
- `CALI`: Caliper
- `RSHA`, `RMED`, `RDEP`, `RMIC`, `RXO`: Resistivity measurements
- `RHOB`: Bulk density
- `GR`, `SGR`: Gamma ray
- `NPHI`: Neutron porosity
- `PEF`: Photoelectric factor
- `DTC`, `DTS`: Sonic velocities
- `SP`: Spontaneous potential
- `BS`: Bit size
- `ROP`, `ROPA`: Rate of penetration
- `DCAL`: Differential caliper
- `DRHO`: Density correction
- `MUDWEIGHT`: Mud weight

### Target Variable
- `FORCE_2020_LITHOFACIES_LITHOLOGY`: Lithology class label (categorical)
- `FORCE_2020_LITHOFACIES_CONFIDENCE`: Confidence score for the label

### Lithology Classes
The target classes include: `65000`, `30000`, `65030`, `70000`, `80000`, `99000`, `70032`, `88000`, `90000`, `74000`, `86000`, `93000`

**Minority classes** (used for augmentation): `74000`, `86000`, `93000`

## Project Structure

```
├── data/
│   ├── CSV_train.csv          # Training data
│   └── CSV_test.csv           # Test data
├── combined.csv               # Combined dataset with predictions
├── Copy_of_Untitled4_combiningdataset(umar).ipynb  # Main notebook
└── README.md
```

## Pipeline Workflow

### 1. Data Loading & Combination
- Loads train and test CSV files (semicolon-separated)
- Handles missing values (`""`, `"NA"`, `"NaN"`, `-999.25`)
- Adds predictions to test set
- Combines datasets with an `_is_train` flag

### 2. Preprocessing
- **Categorical encoding**: LabelEncoder for categorical features (`WELL`, `GROUP`, `FORMATION`)
- **Numerical scaling**: MinMaxScaler for numerical features
- **Missing value imputation**: Mode for categorical, median for numerical

### 3. Data Splitting
- Shuffled split with:
  - ~120,000 samples for test set
  - Remaining data split 80/20 for train/validation

### 4. CTGAN Data Augmentation
Addresses class imbalance by generating synthetic samples for minority classes:

- **Target classes**: `74000`, `86000`, `93000`
- **Target size**: 5,000 samples per minority class
- **CTGAN configuration**: 30 epochs
- Synthetic samples evaluated using:
  - **KS Test (Kolmogorov-Smirnov)**: Compares distributions
  - **Correlation heatmaps**: Compares feature relationships
  - **PCA visualization**: Visual comparison of real vs synthetic data

### 5. Neural Network Models

#### Model 1: MLP with Embeddings
```
Input → Embeddings (categorical) + Numerical Features
     → Linear(256) + BatchNorm + ReLU + Dropout(0.3)
     → Linear(128) + BatchNorm + ReLU + Dropout(0.3)
     → Linear(num_classes)
```

#### Model 2: Deep MLP with Attention & Residual Connections
More advanced architecture featuring:
- **Embedding layers** for categorical features (dim=32)
- **Attention blocks**: Self-attention mechanism for feature refinement
- **Residual blocks**: Skip connections for better gradient flow
- **Architecture**: 512 → 256 → 128 → num_classes

### 6. Training
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (lr=1e-3)
- **Epochs**: 100
- **Batch size**: 1024
- **Device**: CUDA if available, else CPU

## Installation

```bash
pip install pandas scikit-learn torch ctgan matplotlib seaborn scipy
```

## Usage

1. **Place data files** in the `data/` directory:
   - `CSV_train.csv.zip`
   - `CSV_test.csv.zip`

2. **Run the notebook** sequentially:
   ```bash
   jupyter notebook Copy_of_Untitled4_combiningdataset\(umar\).ipynb
   ```

3. **Key outputs**:
   - `combined.csv`: Merged dataset with predictions
   - Trained model weights
   - Classification reports
   - Data quality visualizations (KS tests, correlation heatmaps, PCA plots)

## Key Functions

| Function | Description |
|----------|-------------|
| `prepare_features(df)` | Extracts categorical, numerical features and encoded labels |
| `LithologyDataset` | PyTorch Dataset class for well-log data |
| `MLPWithEmbeddings` | Baseline neural network with embeddings |
| `DeepMLPWithAttention` | Advanced model with attention and residual blocks |
| `AttentionBlock` | Self-attention mechanism |
| `ResidualBlock` | Residual connection block |
| `train_model()` | Training loop with validation |

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Classification Report**: Per-class precision, recall, F1-score
- **Validation Accuracy**: Tracked per epoch

## Notes

- The notebook contains two versions of the model training pipeline:
  - **Version 1**: Uses CTGAN-augmented data with a simpler MLP
  - **Version 2 (NEW)**: Uses a deeper architecture with attention mechanisms
- GPU (T4) is recommended for CTGAN training
- The `synthetic_data_dict` stores synthetic samples per class for evaluation

## Known Issues

- The PCA comparison may fail if NaN values remain after synthetic data generation
- Ensure `combined.median()` is applied only to numerical columns when imputing
- Class labels are treated as numeric values; consider using LabelEncoder for target if class order matters

