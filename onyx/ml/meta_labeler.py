import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

try:
    import xgboost as xgb
except ImportError:
    xgb = None

logger = logging.getLogger(__name__)

class MetaLabeler:
    """
    XGBoost Meta-Labeler that predicts the probability of a positive forward return.
    Acts as a secondary filter over base linear alpha scores.
    """
    def __init__(self, model_path: str = None):
        self.model = None
        if xgb is None:
            logger.error("XGBoost is not installed. MetaLabeler cannot be initialized.")
            return
            
        if model_path:
            self.load_model(model_path)
        else:
            # Initialize with conservative default hyperparameters to prevent overfitting
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,          # Keep shallow to prevent overfitting
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42
            )
            
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """
        Trains the XGBoost classifier.
        """
        if self.model is None:
            raise ValueError("Model is not initialized.")
            
        logger.info(f"Training Meta-Labeler on {len(X_train)} samples...")
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        # Using early stopping to prevent overfitting if validation set is provided
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        logger.info("Training complete.")
        
    def predict_probability(self, X: pd.DataFrame) -> pd.Series:
        """
        Returns the probability of success P(Success | X).
        """
        if self.model is None:
            raise ValueError("Model is not initialized or trained.")
            
        probs = self.model.predict_proba(X)[:, 1] # Get probability of class 1
        return pd.Series(probs, index=X.index)
        
    def filter_alphas(self, base_alphas: pd.Series, features: pd.DataFrame, threshold: float = 0.55) -> pd.Series:
        """
        Filters base alpha scores by zeroing out assets where the Meta-Labeler predicts
        a probability of success lower than the threshold.
        """
        probs = self.predict_probability(features)
        
        # Align indices
        filtered_alphas = base_alphas.copy()
        
        for symbol in filtered_alphas.index:
            if symbol in probs.index:
                p = probs[symbol]
                if p < threshold:
                    logger.debug(f"Meta-Labeler rejecting {symbol}: P(Success) = {p:.2f} < {threshold}")
                    filtered_alphas[symbol] = 0.0
            else:
                # If no features exist, default to 0 to be safe
                filtered_alphas[symbol] = 0.0
                
        num_rejected = (filtered_alphas == 0.0).sum() - (base_alphas == 0.0).sum()
        logger.info(f"Meta-Labeler rejected {num_rejected} low-probability assets out of {len(base_alphas)}")
        
        return filtered_alphas
        
    def save_model(self, path: str):
        if self.model is not None:
            self.model.save_model(path)
            logger.info(f"Model saved to {path}")
            
    def load_model(self, path: str):
        if self.model is not None:
            self.model.load_model(path)
            logger.info(f"Model loaded from {path}")
