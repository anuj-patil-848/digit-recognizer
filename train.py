# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-29T17:47:47.080024Z","iopub.execute_input":"2024-10-29T17:47:47.080530Z","iopub.status.idle":"2024-10-29T17:47:47.544311Z","shell.execute_reply.started":"2024-10-29T17:47:47.080463Z","shell.execute_reply":"2024-10-29T17:47:47.543040Z"}}
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt

# %% [code] {"execution":{"iopub.status.busy":"2024-10-29T18:04:04.355873Z","iopub.execute_input":"2024-10-29T18:04:04.356326Z","iopub.status.idle":"2024-10-29T18:04:08.195183Z","shell.execute_reply.started":"2024-10-29T18:04:04.356284Z","shell.execute_reply":"2024-10-29T18:04:08.193976Z"}}
# Load and preprocess data
data = pd.read_csv('/Users/anujpatil/Desktop/digit recognizer/training_data/mnist_train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)
numSamples = len(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

# %% [code] {"execution":{"iopub.status.busy":"2024-10-29T17:47:52.709029Z","iopub.execute_input":"2024-10-29T17:47:52.709384Z","iopub.status.idle":"2024-10-29T17:47:52.717090Z","shell.execute_reply.started":"2024-10-29T17:47:52.709347Z","shell.execute_reply":"2024-10-29T17:47:52.715863Z"}}
# Initialize parameters
def init_params():
    W1 = np.random.randn(128, 784) * np.sqrt(1. / 784)
    b1 = np.zeros((128, 1))
    W2 = np.random.randn(64, 128) * np.sqrt(1. / 128)
    b2 = np.zeros((64, 1))
    W3 = np.random.randn(10, 64) * np.sqrt(1. / 64)
    b3 = np.zeros((10, 1))
    return W1, b1, W2, b2, W3, b3

# %% [code] {"execution":{"iopub.status.busy":"2024-10-29T17:47:52.720501Z","iopub.execute_input":"2024-10-29T17:47:52.721125Z","iopub.status.idle":"2024-10-29T17:47:52.733486Z","shell.execute_reply.started":"2024-10-29T17:47:52.721068Z","shell.execute_reply":"2024-10-29T17:47:52.732132Z"}}
# Activation functions
def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # More numerically stable softmax
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-29T17:47:52.735032Z","iopub.execute_input":"2024-10-29T17:47:52.735436Z","iopub.status.idle":"2024-10-29T17:47:52.745557Z","shell.execute_reply.started":"2024-10-29T17:47:52.735394Z","shell.execute_reply":"2024-10-29T17:47:52.744332Z"}}
# Forward propagation with additional layers
def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# %% [code] {"execution":{"iopub.status.busy":"2024-10-29T17:47:52.747240Z","iopub.execute_input":"2024-10-29T17:47:52.747683Z","iopub.status.idle":"2024-10-29T17:47:52.757582Z","shell.execute_reply.started":"2024-10-29T17:47:52.747635Z","shell.execute_reply":"2024-10-29T17:47:52.756030Z"}}
# One-hot encoding
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Derivative of ReLU
def deriv_ReLU(Z):
    return Z > 0

# %% [code] {"execution":{"iopub.status.busy":"2024-10-29T17:49:01.919670Z","iopub.execute_input":"2024-10-29T17:49:01.920149Z","iopub.status.idle":"2024-10-29T17:49:01.931360Z","shell.execute_reply.started":"2024-10-29T17:49:01.920090Z","shell.execute_reply":"2024-10-29T17:49:01.929824Z"}}
# Prediction functions
def get_predictions(A3):
    return np.argmax(A3, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Make predictions
def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions

# Test prediction
def test_prediction(index, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    
    print(f"Index: {index}")
    print(f"Prediction: {prediction}")
    print(f"Label: {label}")
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-29T17:47:52.759066Z","iopub.execute_input":"2024-10-29T17:47:52.759477Z","iopub.status.idle":"2024-10-29T17:47:52.777062Z","shell.execute_reply.started":"2024-10-29T17:47:52.759430Z","shell.execute_reply":"2024-10-29T17:47:52.775895Z"}}
#Optimizer
def optimizer(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, vW1, vb1, vW2, vb2, vW3, vb3, sW1, sb1, sW2, sb2, sW3, sb3, t, alpha, beta1=0.9, beta2=0.999, epsilon=1e-8):
    vW1 = beta1 * vW1 + (1 - beta1) * dW1
    vb1 = beta1 * vb1 + (1 - beta1) * db1
    vW2 = beta1 * vW2 + (1 - beta1) * dW2
    vb2 = beta1 * vb2 + (1 - beta1) * db2
    vW3 = beta1 * vW3 + (1 - beta1) * dW3
    vb3 = beta1 * vb3 + (1 - beta1) * db3

    sW1 = beta2 * sW1 + (1 - beta2) * (dW1 ** 2)
    sb1 = beta2 * sb1 + (1 - beta2) * (db1 ** 2)
    sW2 = beta2 * sW2 + (1 - beta2) * (dW2 ** 2)
    sb2 = beta2 * sb2 + (1 - beta2) * (db2 ** 2)
    sW3 = beta2 * sW3 + (1 - beta2) * (dW3 ** 2)
    sb3 = beta2 * sb3 + (1 - beta2) * (db3 ** 2)

    vW1_corrected = vW1 / (1 - beta1 ** t)
    vb1_corrected = vb1 / (1 - beta1 ** t)
    vW2_corrected = vW2 / (1 - beta1 ** t)
    vb2_corrected = vb2 / (1 - beta1 ** t)
    vW3_corrected = vW3 / (1 - beta1 ** t)
    vb3_corrected = vb3 / (1 - beta1 ** t)

    sW1_corrected = sW1 / (1 - beta2 ** t)
    sb1_corrected = sb1 / (1 - beta2 ** t)
    sW2_corrected = sW2 / (1 - beta2 ** t)
    sb2_corrected = sb2 / (1 - beta2 ** t)
    sW3_corrected = sW3 / (1 - beta2 ** t)
    sb3_corrected = sb3 / (1 - beta2 ** t)

    W1 -= alpha * vW1_corrected / (np.sqrt(sW1_corrected) + epsilon)
    b1 -= alpha * vb1_corrected / (np.sqrt(sb1_corrected) + epsilon)
    W2 -= alpha * vW2_corrected / (np.sqrt(sW2_corrected) + epsilon)
    b2 -= alpha * vb2_corrected / (np.sqrt(sb2_corrected) + epsilon)
    W3 -= alpha * vW3_corrected / (np.sqrt(sW3_corrected) + epsilon)
    b3 -= alpha * vb3_corrected / (np.sqrt(sb3_corrected) + epsilon)

    return W1, b1, W2, b2, W3, b3, vW1, vb1, vW2, vb2, vW3, vb3, sW1, sb1, sW2, sb2, sW3, sb3

# %% [code] {"execution":{"iopub.status.busy":"2024-10-29T17:47:52.778596Z","iopub.execute_input":"2024-10-29T17:47:52.779091Z","iopub.status.idle":"2024-10-29T17:47:52.797476Z","shell.execute_reply.started":"2024-10-29T17:47:52.779048Z","shell.execute_reply":"2024-10-29T17:47:52.796101Z"}}
# Gradient descent
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3 = init_params()
    
    # Initialize parameters for optimizer
    vW1, vb1, vW2, vb2, vW3, vb3 = np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros_like(b2), np.zeros_like(W3), np.zeros_like(b3)
    sW1, sb1, sW2, sb2, sW3, sb3 = np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros_like(b2), np.zeros_like(W3), np.zeros_like(b3)
    
    for t in range(1, iterations + 1):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        
        # Backpropagation
        one_hot_Y = one_hot(Y)
        dZ3 = A3 - one_hot_Y
        dW3 = 1 / m_train * dZ3.dot(A2.T)
        db3 = 1 / m_train * np.sum(dZ3, axis=1, keepdims=True)
        
        dZ2 = W3.T.dot(dZ3) * deriv_ReLU(Z2)
        dW2 = 1 / m_train * dZ2.dot(A1.T)
        db2 = 1 / m_train * np.sum(dZ2, axis=1, keepdims=True)
        
        dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
        dW1 = 1 / m_train * dZ1.dot(X.T)
        db1 = 1 / m_train * np.sum(dZ1, axis=1, keepdims=True)

        # Update weights and biases using optimizer
        W1, b1, W2, b2, W3, b3, vW1, vb1, vW2, vb2, vW3, vb3, sW1, sb1, sW2, sb2, sW3, sb3 = optimizer(
            W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, vW1, vb1, vW2, vb2, vW3, vb3, sW1, sb1, sW2, sb2, sW3, sb3, t, alpha
        )
        
        if t % 10 == 0:
            predictions = get_predictions(A3)
            accuracy = get_accuracy(predictions, Y)
            if t < 100:
                print(f"Iteration: {t}  | Accuracy: {accuracy * 100:.3f}%")
            else:
                print(f"Iteration: {t} | Accuracy: {accuracy * 100:.3f}%")
    return W1, b1, W2, b2, W3, b3

def train_model(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3 = gradient_descent(X, Y, alpha, iterations)
    
    # Store the trained parameters in a dictionary
    trained_model = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
        'W3': W3,
        'b3': b3
    }
    return trained_model

# %% [code] {"execution":{"iopub.status.busy":"2024-10-29T17:49:08.451482Z","iopub.execute_input":"2024-10-29T17:49:08.451945Z","iopub.status.idle":"2024-10-29T17:53:46.385785Z","shell.execute_reply.started":"2024-10-29T17:49:08.451900Z","shell.execute_reply":"2024-10-29T17:53:46.384485Z"}}
# Train the model
#W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 0.001, 500)
model = train_model(X_train, Y_train, 0.001, 1000)

#using pickle to store trained model as trained_model.pkl and use trained data in app.py
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)

#model = train_model(X_train, Y_train, 0.001, 30)
#test_prediction(10, W1, b1, W2, b2, W3, b3)
#print(model)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-29T18:53:59.271120Z","iopub.execute_input":"2024-10-29T18:53:59.273330Z","iopub.status.idle":"2024-10-29T18:53:59.552835Z","shell.execute_reply.started":"2024-10-29T18:53:59.273223Z","shell.execute_reply":"2024-10-29T18:53:59.551360Z"}}
# Test a sample prediction

#test_prediction(np.random.randint(0,numSamples-1), W1, b1, W2, b2, W3, b3)
