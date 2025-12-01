import numpy as np
import itertools
# An array of 0s and 1s
#Shape: (2,2) -> 2 rows and 2 columns
#1.0 white, while 0.0 is black
combinations =  list(itertools.product([0,1], repeat=4))
x_raw = np.array(combinations)

print(f"Input: {x_raw.shape}")

y = np.where(np.sum(x_raw, axis=1) >= 2, 1, 0)
print(y)

# adding bias neuron
rows = x_raw.shape[0]
print(rows)
bias = np.ones((rows,1))
X = np.hstack((bias, x_raw))
print(X)

# The Current shape of X is now (16,5) -> [Bias, input1 , input2 , input3, input4]

weights = np.zeros(5)
learning_rate = 0.1
loop_through = 40

print("Start the training process")

# We have 16 total samples.
# Let's use the first 12 for TRAINING.
# Let's use the last 4 for TESTING (The AI will never see these during the loop).

n_train = 12

# Slicing the arrays
X_train = X[:n_train]   # First 12 rows
y_train = y[:n_train]   # First 12 targets

X_test  = X[n_train:]   # Last 4 rows
y_test  = y[n_train:]   # Last 4 targets

total_error = 0 # Tracking error count
for i in range(loop_through):
    error = 0
    for j in range(len(X_train)):
        features = X_train[j]
        print(features)
        target = y_train[j]

        weights_sum = np.dot(features, weights)
        prediction = 1 if weights_sum >= 0 else 0

        # calculate the error
        error = target - prediction

        if error != 0:
            weights = weights +(learning_rate * error * features)
            total_error += 1
    if total_error == 0:
        print(f"Training completed in {i+1} iterations.")
        break
print(f"Final weights: {weights}")

# Testing
print("\nTesting...")
for j in range(len(X_test)):
    features = X_test[j]
    true_label = y_test[j]

    # Run the training weights on test
    weights_sum = np.dot(features, weights)
    prediction_ = 1 if weights_sum >= 0 else 0

    status = "Correct" if prediction_ == true_label else "Incorrect"
    print(f" Input: {features[1:]} -> Prediction: {prediction_} (Target: {true_label} [{status}])")


