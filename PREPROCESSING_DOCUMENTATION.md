# Data Preparation and Preprocessing Documentation

## Overview

This document describes the data preparation and preprocessing pipeline implemented for the Food-101 image recognition project. The pipeline ensures clean, normalized, and properly split data ready for model training and reproducible fine-tuning.

## Pipeline Components

### 1. Data Loading (`data_loader.py`)

The `DataLoader` class handles loading the Food-101 dataset from Hugging Face and provides utilities for inspecting the dataset.

**Key Features:**
- Loads Food-101 dataset using `datasets.load_dataset()`
- Displays dataset statistics (number of samples, features, class distribution)
- Shows example entries for inspection

**Usage:**
```python
from data_loader import DataLoader

loader = DataLoader(dataset_name="food101")
dataset = loader.load_data()
loader.display_info()
loader.show_examples(split="train", num_examples=3)
```

### 2. Image Preprocessing (`preprocessing.py`)

The `ImagePreprocessor` class handles all image cleaning, normalization, and augmentation operations.

**Preprocessing Steps:**

#### a. Image Cleaning
- **RGB Conversion**: Ensures all images are in RGB format
- **Corruption Detection**: Identifies and handles corrupted or invalid images
- **Size Validation**: Checks that images have valid dimensions

#### b. Image Normalization
- **Resizing**: Standardizes all images to 224×224 pixels
- **Pixel Normalization**: Normalizes pixel values using ImageNet statistics
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- **Tensor Conversion**: Converts images to PyTorch tensors

#### c. Data Augmentation (Training Only)
- **Random Resized Crop**: Crops random portions of images
- **Random Horizontal Flip**: Flips images horizontally with 50% probability
- **Random Rotation**: Rotates images up to ±15 degrees
- **Color Jitter**: Randomly adjusts brightness, contrast, and saturation

**Usage:**
```python
from preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor(target_size=(224, 224))

# For training (with augmentation)
train_tensor = preprocessor.preprocess_image(image, augment=True)

# For validation/test (without augmentation)
val_tensor = preprocessor.preprocess_image(image, augment=False)
```

### 3. Data Splitting (`data_splitting.py`)

The `DataSplitter` class creates reproducible train/validation/test splits with stratification.

**Split Configuration:**
- Train: 80% of data
- Validation: 10% of data
- Test: 10% of data
- Random seed: 42 (for reproducibility)

**Key Features:**
- **Stratified Splitting**: Maintains class distribution across splits
- **Class Weight Computation**: Calculates weights for handling class imbalance
- **Split Verification**: Validates stratification quality

**Usage:**
```python
from data_splitting import DataSplitter, compute_class_weights

splitter = DataSplitter(random_seed=42)
splits = splitter.split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
splitter.display_split_info(splits, label_names=label_names)

# Compute class weights for imbalanced classes
class_weights = compute_class_weights(splits['train']['labels'])
```

### 4. Complete Pipeline (`pipeline.py`)

The `pipeline.py` script demonstrates the complete end-to-end workflow:

1. Loads the Food-101 dataset
2. Displays dataset information and examples
3. Initializes the preprocessor with target size
4. Validates images for corruption
5. Creates stratified train/val/test splits
6. Computes class weights for imbalance handling
7. Processes and saves example images
8. Saves split information for reproducibility

**Running the Pipeline:**
```bash
python pipeline.py
```

**Outputs:**
- `preprocessed_examples/`: Directory containing original, preprocessed, and augmented example images
- `split_info.json`: JSON file with split indices and dataset statistics

## Preprocessing Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Target Size | 224×224 | Standard input size for most vision models (ResNet, VGG, etc.) |
| Normalization Mean | [0.485, 0.456, 0.406] | ImageNet statistics, standard for transfer learning |
| Normalization Std | [0.229, 0.224, 0.225] | ImageNet statistics, standard for transfer learning |
| Random Seed | 42 | Fixed seed ensures reproducible splits |
| Train Ratio | 0.8 | 80% for training provides sufficient data |
| Val Ratio | 0.1 | 10% for validation provides good performance estimates |
| Test Ratio | 0.1 | 10% for final testing maintains held-out evaluation set |

## Dataset Statistics

**Food-101 Dataset:**
- Total images: 101,000
- Number of classes: 101 (food categories)
- Images per class: 1,000
- Original image sizes: Variable (resized to 224×224)

**After Splitting:**
- Train: ~80,800 images
- Validation: ~10,100 images
- Test: ~10,100 images

## Handling Dataset Challenges

### 1. Class Balance
- **Issue**: Food-101 is relatively balanced (1,000 images per class)
- **Solution**: Class weights computed for flexibility in handling any imbalance
- **Implementation**: `compute_class_weights()` function in `data_splitting.py`

### 2. Image Quality
- **Issue**: Some images may be corrupted or in different formats
- **Solution**: Corruption detection and RGB conversion
- **Implementation**: `clean_image()` and `is_corrupted()` methods in `ImagePreprocessor`

### 3. Size Variance
- **Issue**: Original images have variable sizes
- **Solution**: Standardized resizing to 224×224
- **Implementation**: `transforms.Resize()` in preprocessing transforms

### 4. Reproducibility
- **Issue**: Need consistent results across runs
- **Solution**: Fixed random seed (42) for all random operations
- **Implementation**: Set in `DataSplitter` constructor

## Example Usage Workflow

```python
# 1. Load data
from data_loader import DataLoader
loader = DataLoader()
dataset = loader.load_data()

# 2. Create splits
from data_splitting import DataSplitter
splitter = DataSplitter(random_seed=42)
splits = splitter.split_dataset(dataset)

# 3. Initialize preprocessor
from preprocessing import ImagePreprocessor
preprocessor = ImagePreprocessor(target_size=(224, 224))

# 4. Get training data
train_data = dataset['train']
train_indices = splits['train']['indices']

# 5. Preprocess images for training
for idx in train_indices[:10]:  # First 10 examples
    image = train_data[idx]['image']
    label = train_data[idx]['label']
    
    # Apply preprocessing with augmentation
    processed = preprocessor.preprocess_image(image, augment=True)
    
    # Use processed tensor for model training
    # model.train(processed, label)
```

## Validation of Preprocessing

The pipeline includes validation steps to ensure correct preprocessing:

1. **Image Corruption Check**: First 100 images checked for corruption
2. **RGB Conversion Check**: Counts images needing format conversion
3. **Stratification Verification**: Validates class distribution similarity across splits
4. **Visual Inspection**: Saves preprocessed examples for manual verification

## Integration with Model Training

To use this pipeline with your model:

1. Run `pipeline.py` to prepare the data
2. Load the split indices from `split_info.json`
3. Create data loaders using the splits and preprocessor
4. Use `augment=True` for training data
5. Use `augment=False` for validation and test data
6. Apply class weights during training if needed

## Requirements

See `requirements.txt` for full dependencies:
- datasets >= 2.14.0 (Hugging Face datasets)
- Pillow >= 10.0.0 (Image processing)
- numpy >= 1.24.0 (Numerical operations)
- scikit-learn >= 1.3.0 (Train/test splitting)
- torch >= 2.0.0 (Deep learning framework)
- torchvision >= 0.15.0 (Vision transforms)
- tqdm >= 4.65.0 (Progress bars)

## Future Enhancements

Potential improvements for future iterations:
1. Support for additional datasets (HuffPost text classification)
2. Advanced augmentation techniques (Mixup, CutMix)
3. Automatic hyperparameter tuning for augmentation
4. Caching preprocessed images for faster training
5. Multi-GPU data loading support
