import pandas as pd
import numpy as np
from onyx.ml.meta_labeler import MetaLabeler

def main():
    print("Initializing C++ Engine...")
    labeler = MetaLabeler()
    
    # Generate some mock financial data
    print("Generating Mock Data...")
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(100, 3), columns=['F1', 'F2', 'F3'])
    # Y is 1 if F1 + F2 > 0 else 0
    y_train = pd.Series((X_train['F1'] + X_train['F2'] > 0).astype(int))
    
    print("Training C++ Engine (1000 epochs)...")
    labeler.train(X_train, y_train, learning_rate=0.05, epochs=1000)
    
    print("\nTrained Weights:", labeler.weights)
    print("Trained Bias:", labeler.bias)
    
    print("\nRunning Inference...")
    probs = labeler.predict_probability(X_train.head(5))
    for i, p in enumerate(probs):
        print(f"Sample {i} Probability of Success: {p:.2%}")

if __name__ == "__main__":
    main()
