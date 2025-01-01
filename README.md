# Reddit Climate Analysis

This project analyzes Reddit discussions about climate change, focusing on distinguishing between posts dedicated to climate science and climate action. The analysis uses both traditional machine learning approaches and modern transformer-based models.

## Project Structure

```
reddit_climate_analysis/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── model_config.yaml
│   └── data_config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── visualization/
│   └── utils/
├── notebooks/
└── tests/
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/reddit_climate_analysis.git
cd reddit_climate_analysis
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data

The project uses data from three subreddits:
- r/climate (labeled as "science")
- r/ClimateOffensive and r/ClimateActionPlan (labeled as "action")

The dataset contains 3,274 posts (1,039 "action" and 2,235 "science" posts).

## Models

The project implements several models:
1. Traditional ML Models:
   - Random Forest
   - Logistic Regression
   - SVM
   - Naive Bayes

2. Transformer Models:
   - BERT
   - XLNet
   - DistilBERT
   - RoBERTa

## Usage

1. Data Preprocessing:
```python
from src.data.preprocessing import TextPreprocessor, MetadataProcessor

# Initialize preprocessors
text_processor = TextPreprocessor()
metadata_processor = MetadataProcessor()

# Process data
processed_df = text_processor.process_dataframe(df)
processed_df = metadata_processor.process_metadata(processed_df)
```

2. Training Models:
```python
from src.models.transformers.bert_model import BERTClassifier
from src.models.traditional.ml_models import TraditionalModels

# Train BERT model
bert_classifier = BERTClassifier()
bert_classifier.train(train_loader)

# Train traditional models
traditional_models = TraditionalModels()
results = traditional_models.train