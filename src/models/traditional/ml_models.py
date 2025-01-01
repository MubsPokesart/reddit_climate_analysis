from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml
from typing import Dict, Any
import numpy as np

class TraditionalModels:
    def __init__(self, config_path: str = 'config/model_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['model_parameters']['traditional']
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=self.config['random_forest']['n_estimators'],
                random_state=self.config['random_forest']['random_state']
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=self.config['logistic_regression']['max_iter']
            ),
            'SVM': LinearSVC(
                max_iter=self.config['svm']['max_iter']
            ),
            'Naive Bayes': MultinomialNB()
        }
        
    def train_and_evaluate(self, X_train: np.ndarray, X_test: np.ndarray, 
                          y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
            
        return results