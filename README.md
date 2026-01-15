# ImageRecognition

A comprehensive image recognition project with complete data preparation and preprocessing pipeline for the Food-101 dataset.

## Project Overview

This project implements data preparation and preprocessing steps for image recognition tasks, specifically designed for the Food-101 dataset. The pipeline includes data loading, cleaning, normalization, augmentation, and stratified splitting for reproducible model training.

## Features

- **Data Loading**: Automated loading of Food-101 dataset from Hugging Face
- **Image Preprocessing**: Cleaning, RGB conversion, resizing, and normalization
- **Data Augmentation**: Training-time augmentation (flip, rotation, color jitter, crop)
- **Stratified Splitting**: Reproducible 80/10/10 train/val/test splits
- **Class Weight Computation**: Handle potential class imbalance
- **Validation Tools**: Corruption detection and stratification verification

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running the Pipeline

```bash
python pipeline.py
```

This will:
1. Download the Food-101 dataset
2. Display dataset statistics
3. Create stratified splits
4. Preprocess sample images
5. Save examples and split information

## Project Structure

```
ImageRecognition/
├── README.md                          # This file
├── PREPROCESSING_DOCUMENTATION.md     # Detailed preprocessing documentation
├── requirements.txt                   # Project dependencies
├── data_loader.py                     # Dataset loading utilities
├── preprocessing.py                   # Image preprocessing and augmentation
├── data_splitting.py                  # Train/val/test splitting with stratification
├── pipeline.py                        # Complete pipeline demonstration
├── preprocessed_examples/             # Example preprocessed images (generated)
└── split_info.json                    # Split indices and metadata (generated)
```

## Module Documentation

### data_loader.py
Handles loading the Food-101 dataset and displaying dataset information.

**Key Classes:**
- `DataLoader`: Main class for loading and inspecting datasets

### preprocessing.py
Implements image cleaning, normalization, and augmentation.

**Key Classes:**
- `ImagePreprocessor`: Handles all preprocessing operations

**Key Features:**
- RGB conversion
- Corruption detection
- ImageNet normalization
- Training augmentation pipeline

### data_splitting.py
Creates reproducible stratified splits with class weight computation.

**Key Classes:**
- `DataSplitter`: Handles train/val/test splitting

**Key Functions:**
- `compute_class_weights()`: Calculate weights for imbalanced classes

### pipeline.py
Demonstrates the complete end-to-end preprocessing workflow.

## Preprocessing Pipeline

1. **Load Dataset**: Download Food-101 from Hugging Face
2. **Validate Images**: Check for corruption and format issues
3. **Clean Images**: Convert to RGB and validate dimensions
4. **Normalize**: Resize to 224×224 and normalize with ImageNet stats
5. **Augment**: Apply data augmentation for training data
6. **Split**: Create stratified 80/10/10 train/val/test splits
7. **Compute Weights**: Calculate class weights for training

## Usage Example

```python
from data_loader import DataLoader
from preprocessing import ImagePreprocessor
from data_splitting import DataSplitter

# Load dataset
loader = DataLoader()
dataset = loader.load_data()

# Create splits
splitter = DataSplitter(random_seed=42)
splits = splitter.split_dataset(dataset)

# Initialize preprocessor
preprocessor = ImagePreprocessor(target_size=(224, 224))

# Preprocess an image
image = dataset['train'][0]['image']
processed_tensor = preprocessor.preprocess_image(image, augment=True)
```

## Configuration

Key parameters (defined in respective modules):
- **Image Size**: 224×224 pixels
- **Normalization**: ImageNet mean and std
- **Random Seed**: 42 (for reproducibility)
- **Split Ratios**: 80% train, 10% validation, 10% test

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Hugging Face datasets
- See `requirements.txt` for complete list

## Documentation

For detailed preprocessing documentation, see [PREPROCESSING_DOCUMENTATION.md](PREPROCESSING_DOCUMENTATION.md).

## Next Steps

After running the preprocessing pipeline:
1. Use the preprocessed splits for model training
2. Apply augmentation (`augment=True`) during training
3. Use class weights for handling any imbalance
4. Evaluate on the held-out test set

## License

This project is for educational purposes.