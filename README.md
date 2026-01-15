# ImageRecognition


Goals
Implement the data preparation and preprocessing steps that you proposed in Milestone 1. You’ll clean, normalize, and split your data so that it’s ready for modeling and reproducible fine-tuning.

Steps to Follow
Load your chosen dataset

Use datasets.load_dataset() from Hugging Face to load Food-101 or HuffPost.
Display basic information (e.g., number of samples, feature names, example entries).
Apply cleaning and normalization

Images:

Ensure all images are in RGB format.
Resize or crop to a consistent shape (e.g., 224 × 224).
Drop or fix any corrupted files.
Text:

Concatenate headline + summary (for HuffPost).
Strip whitespace, convert to lowercase if appropriate, and remove empty samples.
Optionally remove duplicates or extremely short entries.
Standardize or tokenize the inputs

Images:

Normalize pixel values (e.g., divide by 255.0).
Define a minimal augmentation pipeline (e.g., random flip, crop, or rotation).
Text:

Create a tokenizer or TextVectorization layer.
Set a target max_length based on your analysis from Milestone 1 (e.g., 95th percentile).
Apply padding/truncation and build tensors for input + labels.
Handle dataset-specific challenges

If you identified class imbalance, compute label counts and, if needed, create a dictionary of class_weights.
If you noted length or size variance, verify that your truncation or resizing works as intended.
If you planned noise filtering, include the cleaning step and briefly explain your criteria (e.g., remove items with missing text or unreadable images).
Create reproducible splits

Split your cleaned dataset into train, validation, and test subsets (e.g., 80 / 10 / 10).
Use a fixed random seed for reproducibility (random_seed = 42).
Use stratified splits (e.g., with train_test_split and stratify = labels).
Display the size of each subset.
Document your pipeline

Summarize your preprocessing steps clearly in Markdown or code comments.
Save or display a few representative examples after preprocessing to confirm the transformations are correct.
