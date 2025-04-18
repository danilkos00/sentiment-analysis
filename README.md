# Clothing Reviews Sentiment Analysis

## Project Description
This project **analyzes customer reviews** in the clothing category and **predicts their sentiment** — positive or negative.

The model is based on a **fine-tuned version of `distilbert-base-uncased`** from Huggingface Transformers.

## Technologies Used
- **PyTorch**
- **Pandas**
- **NumPy**
- **Huggingface Transformers**
- **scikit-learn**
- **spaCy**
- **NLTK**

## Dataset
The project uses the **Womens Clothing E-Commerce Reviews** dataset.

## Quick Start

You can run the project either in Google Colab or locally.

### Launch via Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danilkos00/sentiment-analysis/blob/main/sentiment.ipynb)

### Local Launch

1. Clone the repository:
   ```bash
   git clone https://github.com/danilkos00/sentiment-analysis.git
   cd sentiment-analysis

2. Install dependencies:
    ```bash
    pip install -r requirements.txt -qq

3. Pretrained model weights are automatically downloaded inside the notebook. Download link for model weights:
    ```bash
    https://drive.google.com/uc?id=1snKee0oLYAKJ-F5sTFZmh7qpEZrNU-Xg

## Project Structure
    sentiment-analysis/
    ├── data/           # Dataset
    ├── src/           # Source code for model and data utilities
    ├── requirements.txt # Project dependencies
    ├── sentiment.ipynb     # Jupyter-notebook
    └── README.md      # Project description

