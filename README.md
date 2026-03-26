# Semantic-Detail Dual-Stream Decoupled HRNet Project

## Project Overview

This project implements a Semantic-Detail Dual-Stream Decoupled HRNet (SDHRNet) model for greenhouse semantic segmentation tasks. The model improves segmentation accuracy for different types of greenhouses by separating the extraction processes of semantic features and detail features.

## Core Files Description

### 1. Data Processing and Analysis

#### select_samples.py
- **Function**: Selects high-quality samples from the dataset for frequency analysis
- **Purpose**: Filters samples containing plastic sheds and solar greenhouses, ensuring sufficient target class coverage
- **Parameters**:
  - `dataset_root`: Dataset root path
  - `output_root`: Output directory
  - `target_count`: Target sample count
  - `area_threshold`: Area threshold to ensure target regions are sufficiently large
- **Output**:
  - `selected_samples_for_fft/plastic_shed`: Plastic shed samples
  - `selected_samples_for_fft/solar_greenhouse`: Solar greenhouse samples

#### analyze_frequency.py
- **Function**: Analyzes frequency characteristics of two greenhouse types
- **Purpose**: Analyzes frequency properties of plastic sheds (high-frequency features) and solar greenhouses (low-frequency features) through FFT and Laplacian variance calculations
- **Parameters**:
  - `data_root`: Input data path
  - `output_dir`: Output directory
  - `PATCH_SIZE`: Patch size
- **Output**:
  - Spectrum images of plastic sheds and solar greenhouses
  - Frequency difference heatmap
  - Radial distribution curve
  - Laplacian variance statistics

### 2. Model Configuration and Training

#### cfg.py
- **Function**: Project basic configuration
- **Configuration Items**:
  - `BATCH_SIZE`: Batch size
  - `EPOCH_NUMBER`: Number of training epochs
  - `DATASET`: Dataset configuration (name and class count)
  - `SEGFORMER_VERSION`: SegFormer model version
  - `crop_size`: Cropping size
  - Dataset paths: Paths to training, validation, and test images and labels

#### train.py
- **Function**: Model training
- **Supported Models**: UNet, Deeplab_v3plus, SegFormer, HRNet, SDHRNet, etc.
- **Training Process**:
  - Early stopping mechanism
  - Learning rate adjustment
  - Gradient clipping
  - Training and validation metric calculation
- **Output**:
  - Model weight files: `./Results/weights/{model_name}_weight/`
  - Training logs and metrics

### 3. Model Testing and Evaluation

#### test.py
- **Function**: Model testing and evaluation
- **Evaluation Metrics**:
  - Intersection over Union (IoU)
  - Pixel accuracy
  - Class accuracy
  - Precision, recall, F1 score
- **Output**:
  - Detailed evaluation metrics: `./Results/metrics/{model_name}_metrics.txt`
  - Confusion matrix: `./Results/confusion_matrix/{model_name}_confusion_matrix.png`

#### predict_with_comparison.py
- **Function**: Model prediction and result comparison
- **Comparison Modes**:
  - `side_by_side`: Side-by-side display of original image, ground truth, and prediction
  - `overlay`: Overlay display
  - `quad_view`: Four-view display (original image, ground truth, prediction, difference map)
- **Output**:
  - Comparison images: `./Results/prediction_comparison/{model_name}_{compare_mode}/`
  - Individual prediction and label images

#### visualize_feature_maps.py
- **Function**: Generate model feature map heatmaps
- **Visualization Content**:
  - Original image
  - Ground truth
  - Prediction result
  - Semantic branch feature map
  - Detail branch feature map
  - Complementarity mechanism display map
- **Output**:
  - Heatmaps: `./Results/feature_maps_jet/`

## Environment Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- OpenCV
- Matplotlib
- Seaborn
- tqdm

## Usage Guide

### 1. Data Preparation

1. Configure the dataset paths in `cfg.py`
2. Run `select_samples.py` to select samples for frequency analysis:
   ```bash
   python select_samples.py
   ```

### 2. Frequency Analysis

Run `analyze_frequency.py` to analyze frequency characteristics of the two greenhouse types:
```bash
python analyze_frequency.py
```

### 3. Model Training

Run `train.py` to train the model, specifying the model type:
```bash
python train.py --model SDHRNet --patience 30 --monitor miou
```

### 4. Model Testing

Run `test.py` to test model performance:
```bash
python test.py --model SDHRNet
```

To test with a specific weight file:
```bash
python test.py --model SDHRNet --weight_path Results\weights\SDHRNet_weight\52.pth
```

### 5. Prediction and Visualization

Run `predict_with_comparison.py` to generate prediction result comparisons:
```bash
python predict_with_comparison.py --model SDHRNet --compare_mode quad_view --num_samples 5
```

To predict with a specific weight file:
```bash
python predict_with_comparison.py --model SDHRNet --weight_path Results\weights\SDHRNet_weight\52.pth --compare_mode quad_view --num_samples 5
```

Run `visualize_feature_maps.py` to generate feature map heatmaps:
```bash
python visualize_feature_maps.py
```

## Result Storage

| Result Type | Storage Path | Description |
|------------|-------------|-------------|
| Model Weights | `./Results/weights/{model_name}_weight/` | Model weights saved during training |
| Evaluation Metrics | `./Results/metrics/{model_name}_metrics.txt` | Detailed model evaluation metrics |
| Confusion Matrix | `./Results/confusion_matrix/{model_name}_confusion_matrix.png` | Confusion matrix of model predictions |
| Prediction Comparison | `./Results/prediction_comparison/{model_name}_{compare_mode}/` | Prediction result comparisons in different modes |
| Feature Map Heatmaps | `./Results/feature_maps_jet/` | Heatmaps of semantic and detail branch features |
| Frequency Analysis Results | `./frequency_analysis_final/` | Greenhouse frequency feature analysis results |

## Model Architecture

The SDHRNet model consists of the following components:
1. HRNet W48 backbone network
2. Multi-scale semantic branch: Uses different-sized convolution kernels to capture semantic features
3. Dilated detail branch: Uses dilated convolution and pooling operations to capture detail features
4. Adaptive fusion module: Dynamically adjusts weights based on feature importance
5. Classification layer: Generates final segmentation results

## Ablation Experiments

To evaluate the contribution of each component in SDHRNet, the project includes ablation experiments implemented in `Models/AblationHRNet.py`:

### 1. HRNetWithSemanticBranch
- **Description**: HRNet backbone with only the multi-scale semantic branch
- **Purpose**: Evaluates the effectiveness of the semantic branch alone
- **Command example**:
  ```bash
  python train.py --model HRNetWithSemanticBranch
  python test.py --model HRNetWithSemanticBranch
  ```

### 2. HRNetWithDetailBranch
- **Description**: HRNet backbone with only the dilated detail branch
- **Purpose**: Evaluates the effectiveness of the detail branch alone
- **Command example**:
  ```bash
  python train.py --model HRNetWithDetailBranch
  python test.py --model HRNetWithDetailBranch
  ```

### 3. HRNetWithSimpleFusion
- **Description**: HRNet backbone with both semantic and detail branches, but using simple addition fusion instead of adaptive fusion
- **Purpose**: Evaluates the impact of the adaptive fusion mechanism
- **Command example**:
  ```bash
  python train.py --model HRNetWithSimpleFusion
  python test.py --model HRNetWithSimpleFusion
  ```

## Technical Features

1. **Dual-Stream Decoupling**: Separates semantic and detail feature extraction for separate optimization
2. **Multi-scale Features**: Utilizes HRNet's multi-scale feature representation capability
3. **Adaptive Fusion**: Dynamically adjusts fusion weights based on feature importance in different regions
4. **Frequency Awareness**: Designs network structure based on frequency analysis results, optimizing for the feature characteristics of different greenhouse types

## Notes

1. Ensure the dataset is organized according to the specified format
2. Adjust `BATCH_SIZE` according to hardware conditions before training
3. If encountering memory issues during testing, try using `--batch_size 1`
4. Generated heatmaps and comparison images can be directly used in papers or reports

## License

This project is for academic research use only.