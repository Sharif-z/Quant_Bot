import os
import logging
import pandas as pd
import numpy as np
from onyx.ml.meta_labeler import MetaLabeler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_mock_ml_data(samples=5000):
    """
    Generates synthetic feature data for the Meta-Labeler since we don't have a 
    fully hydrated multi-year Point-in-Time SQLite database populated in this environment.
    Features: Quality, Value, Momentum, Volatility, Marketaux_Sentiment, News_Volume.
    Target: 1 if Forward 10-day Return > 0, else 0.
    """
    np.random.seed(42)
    
    # Generate random features
    quality = np.random.uniform(0, 10, samples)
    value = np.random.uniform(0, 10, samples)
    momentum = np.random.uniform(-5, 15, samples)
    volatility = np.random.uniform(0.01, 0.10, samples)
    marketaux_sentiment = np.random.uniform(-1, 1, samples)
    news_volume = np.random.poisson(lam=5, size=samples)
    
    # Introduce non-linear complex interactions to simulate a real market
    # e.g., High momentum is good, UNLESS volatility is also extremely high (bubble burst risk)
    # High quality + high value is excellent.
    # PHASE 13 Addition: High momentum but terrible sentiment = crash risk
    
    latent_return = (quality * 0.5) + (value * 0.3) + (momentum * 0.4)
    # Penalize high vol if momentum is high
    latent_return -= np.where((momentum > 10) & (volatility > 0.07), 15.0, 0)
    # Penalize momentum if news sentiment is extremely negative
    latent_return -= np.where((momentum > 8) & (marketaux_sentiment < -0.5), 20.0, 0)
    # Add random market noise
    latent_return += np.random.normal(0, 3, samples)
    
    # Target label: 1 if latent return > median, else 0
    median_return = np.median(latent_return)
    targets = (latent_return > median_return).astype(int)
    
    df = pd.DataFrame({
        'quality': quality,
        'value': value,
        'momentum': momentum,
        'volatility': volatility,
        'marketaux_sentiment': marketaux_sentiment,
        'news_volume': news_volume,
        'target': targets
    })
    
    return df

def train_and_evaluate():
    logger.info("Initializing Meta-Labeler Training Pipeline...")
    
    # 1. Load Data
    df = generate_mock_ml_data()
    
    # 2. Chronological Train/Test Split (Purged)
    # We simulate chronological split by just splitting arrays since data is mock, 
    # but in production, we strictly split by Date to avoid lookahead bias.
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    features = ['quality', 'value', 'momentum', 'volatility', 'marketaux_sentiment', 'news_volume']
    
    X_train = train_df[features]
    y_train = train_df['target']
    
    X_test = test_df[features]
    y_test = test_df['target']
    
    logger.info(f"Train Set: {len(X_train)} | Test Set: {len(X_test)}")
    
    # 3. Train Model
    labeler = MetaLabeler()
    labeler.train(X_train, y_train, X_test, y_test)
    
    # 4. Evaluate the Filter's Edge
    probs = labeler.predict_probability(X_test)
    
    # Evaluate Accuracy on high-probability trades (Threshold > 0.60)
    high_prob_mask = probs > 0.60
    if high_prob_mask.sum() > 0:
        win_rate = y_test[high_prob_mask].mean()
        logger.info(f"Trades with P(Success) > 60%: {high_prob_mask.sum()}")
        logger.info(f"Win Rate on High-Prob Trades: {win_rate:.2%}")
    else:
        logger.warning("No trades met the 60% probability threshold.")
        
    # Contrast with low probability trades
    low_prob_mask = probs < 0.40
    if low_prob_mask.sum() > 0:
        low_win_rate = y_test[low_prob_mask].mean()
        logger.info(f"Trades with P(Success) < 40%: {low_prob_mask.sum()}")
        logger.info(f"Win Rate on Low-Prob Trades: {low_win_rate:.2%}")

if __name__ == "__main__":
    train_and_evaluate()
