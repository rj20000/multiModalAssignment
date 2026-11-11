# Multimodal Sentiment Classification of Internet Memes (Memotion 7k)

This project classifies internet memes into Positive, Neutral, and Negative sentiment using frozen CLIP features and compares early fusion, late fusion, and unimodal baselines.

## Dataset
Download the Memotion 7k dataset from Kaggle:  
https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k

## Setup Instructions

### 1. Upload the Dataset to Google Drive
- Download the dataset zip file from Kaggle.
- Upload the zip file (`memotion_dataset_7k.zip`) to your Google Drive.

### 2. Open the Notebook in Google Colab
- Upload the notebook `a1947703_raynouldJosephAssignmentFile.ipynb` to Colab, or open it directly from GitHub.
- Go to **Runtime → Change runtime type → GPU** (recommended).

### 3. Mount Google Drive and Unzip the Dataset
Run the following cells in order:

```python
from google.colab import drive
drive.mount('/content/drive')


# Unzip the dataset (run only once)
!unzip -q "/content/drive/MyDrive/memotion_dataset_7k.zip" -d "/content/drive/MyDrive/"




# Set the data path
DATA_DIR = "/content/drive/MyDrive/memotion_dataset_7k"
print("Dataset ready:", DATA_DIR)


4. Run the Notebook

Run all cells from top to bottom.
The notebook will:
Load and preprocess the data
Extract frozen CLIP features
Train Image-only, Text-only, Early Fusion, and Late Fusion models
Evaluate and plot results


Requirements
The notebook is designed for Google Colab. All required libraries (PyTorch, Transformers, scikit-learn, etc.) are pre-installed in Colab.
Results
Late Fusion achieved the highest Macro F1 (≈0.35) on the test set. All models struggled with the minority Negative class due to class imbalance.
Reproducibility



Stratified 70/15/15 split
Frozen openai/clip-vit-base-patch32 for feature extraction
Early stopping based on validation Macro F1
Class-weighted loss to handle imbalance
