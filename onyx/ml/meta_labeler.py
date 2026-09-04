import numpy as np
import pandas as pd
import logging
import ctypes
import os
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Determine the absolute path to the compiled shared object
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(BASE_DIR, "engine", "libonyx_ml.so")

class MetaLabeler:
    """
    Custom C++ Meta-Labeler that predicts the probability of a positive forward return.
    Replaces XGBoost with a raw C++ Logistic Regression Engine for ultra-low latency.
    """
    def __init__(self, model_path: str = None):
        self.weights = None
        self.bias = 0.0
        self.features_count = 0
        self.c_engine = None
        
        try:
            self._init_ctypes()
        except Exception as e:
            logger.error(f"Failed to load C++ ML Engine: {e}")
            raise
            
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
    def _init_ctypes(self):
        """Binds the C++ shared object library to Python."""
        if not os.path.exists(LIB_PATH):
            logger.warning(f"Missing C++ engine at {LIB_PATH}. Attempting to auto-compile...")
            cpp_path = os.path.join(BASE_DIR, "engine", "onyx_ml.cpp")
            
            # Try clang++ (Termux/Mac) or g++ (Linux/Render)
            compile_cmd = f"clang++ -shared -fPIC -O3 {cpp_path} -o {LIB_PATH} || g++ -shared -fPIC -O3 {cpp_path} -o {LIB_PATH}"
            os.system(compile_cmd)
            
            if not os.path.exists(LIB_PATH):
                raise FileNotFoundError(f"Failed to auto-compile C++ engine. Please compile manually.")
                
            logger.info("Auto-compilation successful!")
            
        self.c_engine = ctypes.CDLL(LIB_PATH)
        
        # void train(double* X, double* y, double* weights, double* bias, int rows, int cols, double lr, int epochs)
        self.c_engine.train.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int
        ]
        
        # void predict(double* X, double* weights, double bias, double* predictions, int rows, int cols)
        self.c_engine.predict.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            ctypes.c_int,
            ctypes.c_int
        ]
            
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, learning_rate=0.01, epochs=1000):
        """
        Trains the C++ Logistic Regression model.
        """
        logger.info(f"Training C++ Meta-Labeler on {len(X_train)} samples for {epochs} epochs...")
        
        rows, cols = X_train.shape
        self.features_count = cols
        
        # If not already initialized, initialize weights to 0
        if self.weights is None or len(self.weights) != cols:
            self.weights = np.zeros(cols, dtype=np.float64)
            self.bias = 0.0
            
        X_c = np.ascontiguousarray(X_train.values, dtype=np.float64)
        y_c = np.ascontiguousarray(y_train.values, dtype=np.float64)
        w_c = np.ascontiguousarray(self.weights, dtype=np.float64)
        b_c = ctypes.c_double(self.bias)
        
        # Execute C++ training
        self.c_engine.train(
            X_c, 
            y_c, 
            w_c, 
            ctypes.byref(b_c), 
            ctypes.c_int(rows), 
            ctypes.c_int(cols), 
            ctypes.c_double(learning_rate), 
            ctypes.c_int(epochs)
        )
        
        # Retrieve trained parameters
        self.weights = w_c
        self.bias = b_c.value
        
        logger.info(f"Training complete. C++ Engine Bias: {self.bias:.4f}")
        
    def predict_probability(self, X: pd.DataFrame) -> pd.Series:
        """
        Returns the probability of success P(Success | X).
        """
        if self.weights is None:
            raise ValueError("Model is not initialized or trained.")
            
        rows, cols = X.shape
        if cols != self.features_count:
            raise ValueError(f"Feature shape mismatch. Expected {self.features_count}, got {cols}")
            
        X_c = np.ascontiguousarray(X.values, dtype=np.float64)
        w_c = np.ascontiguousarray(self.weights, dtype=np.float64)
        preds_c = np.zeros(rows, dtype=np.float64)
        
        # Execute C++ inference
        self.c_engine.predict(
            X_c,
            w_c,
            ctypes.c_double(self.bias),
            preds_c,
            ctypes.c_int(rows),
            ctypes.c_int(cols)
        )
        
        return pd.Series(preds_c, index=X.index)
        
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
                filtered_alphas[symbol] = 0.0
                
        num_rejected = (filtered_alphas == 0.0).sum() - (base_alphas == 0.0).sum()
        logger.info(f"C++ Engine rejected {num_rejected} low-probability assets out of {len(base_alphas)}")
        
        return filtered_alphas
        
    def save_model(self, path: str):
        if self.weights is not None:
            data = {
                "weights": self.weights.tolist(),
                "bias": self.bias,
                "features_count": self.features_count
            }
            with open(path, 'w') as f:
                json.dump(data, f)
            logger.info(f"C++ Model state saved to {path}")
            
    def load_model(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
            self.weights = np.array(data["weights"], dtype=np.float64)
            self.bias = float(data["bias"])
            self.features_count = int(data["features_count"])
        logger.info(f"C++ Model state loaded from {path}")
