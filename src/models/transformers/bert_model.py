import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import yaml
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.preprocessing import LabelEncoder

class SubredditDataset(Dataset):
    def __init__(self, texts: np.ndarray, labels: np.ndarray, tokenizer: BertTokenizer, max_len: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class BERTClassifier:
    def __init__(self, config_path: str = 'config/model_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['model_parameters']['bert']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(self.config['pretrained_model'])
        self.model = None
        self.label_encoder = LabelEncoder()

    def prepare_data(self, texts: np.ndarray, labels: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Create datasets
        dataset = SubredditDataset(
            texts, 
            encoded_labels, 
            self.tokenizer, 
            self.config['max_length']
        )
        
        return DataLoader(
            dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )

    def train(self, train_loader: DataLoader) -> None:
        self.model = BertForSequenceClassification.from_pretrained(
            self.config['pretrained_model'],
            num_labels=len(self.label_encoder.classes_)
        ).to(self.device)

        optimizer = AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * self.config['num_epochs']
        )

        for epoch in range(self.config['num_epochs']):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{self.config['num_epochs']}, Loss: {avg_loss:.4f}")

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())

        return self.label_encoder.inverse_transform(predictions)