import numpy as np


# Activation Function
def sigmoid(w_sum):
    return 1 / (1 + np.exp(-w_sum))


# Get Prediction
def predict(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)


# Loss Function
def cross_entropy(target, pred):
    return -(target * np.log10(pred) + (1 - target) * (np.log10(1 - pred)))


# Update Weights
def gradient_descent(x, y, weights, bias, learn_rate, pred):
    new_weights = []
    bias += learn_rate * (y - pred)

    for w, xi in zip(weights, x):
        new_weight = w + learn_rate * (y - pred) * xi
        new_weights.append(new_weight)
    return new_weights, bias


# Data
features = np.array(([0.1, 0.5, 0.2], [0.2, 0.3, 0.1], [0.7, 0.4, 0.2], [0.1, 0.4, 0.3]))
targets = np.array([0, 1, 0, 1])

epochs = 10
learn_rate = 0.1

errors = []
weights = np.array([0.4, 0.2, 0.6])
bias = 0.5

new_weights = []

for e in range(epochs):
    for x, y in zip(features, targets):
        pred = predict(x, weights, bias)
        error = cross_entropy(y, pred)
        weights, bias = gradient_descent(x, y, weights, bias, learn_rate, pred)

    # Printing out the log-loss error on the training set
    out = predict(features, weights, bias)
    loss = np.mean(cross_entropy(targets, out))
    errors.append(loss)
    print("\n========== Epoch", e, "==========")
    print("Average loss: ", loss)