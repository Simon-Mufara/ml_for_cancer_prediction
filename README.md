# Bioinformatics ML Cancer Prediction

A machine learning pipeline for cancer type prediction from gene expression data.

## Project Structure

```
bioinformatics-ml-cancer-prediction/
├── data/
│   └── cancer_gene_expression.csv   # Gene expression dataset
├── notebooks/
│   └── exploration.ipynb            # EDA notebook
├── src/
│   ├── train_model.py               # Model training script
│   └── predict.py                   # Inference script
├── app/
│   └── app.py                       # Flask REST API
├── model.pkl                        # Saved model (generated after training)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Train the model

```bash
cd src
python train_model.py
```

### Run predictions

```bash
cd src
python predict.py
```

### Start the API server

```bash
cd app
python app.py
```

Then send a POST request to `http://localhost:5000/predict` with:

```json
{
  "features": [[val1, val2, ...]]
}
```
