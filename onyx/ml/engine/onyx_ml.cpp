#include <cmath>

extern "C" {

    // Helper function for Sigmoid activation
    double sigmoid(double z) {
        return 1.0 / (1.0 + exp(-z));
    }

    // Inference (Prediction) Engine
    // Uses raw pointers to accept Python NumPy arrays directly in memory
    void predict(double* X, double* weights, double bias, double* predictions, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            double z = bias;
            for (int j = 0; j < cols; j++) {
                z += X[i * cols + j] * weights[j];
            }
            predictions[i] = sigmoid(z);
        }
    }

    // Gradient Descent Training Engine
    void train(double* X, double* y, double* weights, double* bias, int rows, int cols, double lr, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double bias_gradient = 0.0;
            
            // Temporary array for weight gradients initialized to 0
            double* weight_gradients = new double[cols](); 

            for (int i = 0; i < rows; i++) {
                double z = *bias;
                for (int j = 0; j < cols; j++) {
                    z += X[i * cols + j] * weights[j];
                }
                
                double prediction = sigmoid(z);
                double error = prediction - y[i];

                bias_gradient += error;
                for (int j = 0; j < cols; j++) {
                    weight_gradients[j] += error * X[i * cols + j];
                }
            }

            // Apply gradients
            *bias -= lr * (bias_gradient / rows);
            for (int j = 0; j < cols; j++) {
                weights[j] -= lr * (weight_gradients[j] / rows);
            }
            
            delete[] weight_gradients; // Prevent memory leaks
        }
    }
}
