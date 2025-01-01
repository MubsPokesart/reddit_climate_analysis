import re
import pandas as pd
import yaml
from typing import List, Dict, Any
import numpy as np
from sklearn.preprocessing import StandardScaler

class TextPreprocessor:
    def __init__(self, config_path: str = 'config/data_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def preprocess_text(self, text: str) -> str:
        if pd.isna(text) or text is None:
            return ''
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        return ' '.join(text.split())
    
    def combine_text_fields(self, row: pd.Series) -> str:
        fields = [
            str(row['post_title']) if not pd.isna(row['post_title']) else '',
            str(row['post_self_text']) if not pd.isna(row['post_self_text']) else '',
            str(row['self_text']) if not pd.isna(row['self_text']) else ''
        ]
        return ' '.join(field for field in fields if field)
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df['combined_text'] = df.apply(self.combine_text_fields, axis=1)
        df['combined_text'] = df['combined_text'].apply(self.preprocess_text)
        return df[df['combined_text'].str.len() > 0].copy()

class MetadataProcessor:
    def __init__(self, config_path: str = 'config/data_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.scaler = StandardScaler()
            
    def process_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        metadata_features = self.config['features']['metadata']
        
        # Fill missing values with median
        for feature in metadata_features:
            df[feature] = df[feature].fillna(df[feature].median())
            
        # Scale features
        df[metadata_features] = self.scaler.fit_transform(df[metadata_features])
        
        return df

class TemporalProcessor:
    def __init__(self, config_path: str = 'config/data_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def process_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        temporal_features = self.config['features']['temporal']
        
        for feature in temporal_features:
            df[feature] = pd.to_datetime(df[feature])
            df[f'{feature}_hour'] = df[feature].dt.hour
            df[f'{feature}_day'] = df[feature].dt.dayofweek
            df[f'{feature}_month'] = df[feature].dt.month
            
        return df