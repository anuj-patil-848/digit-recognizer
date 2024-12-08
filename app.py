import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app, resources={"/predict": {"origins": "*"}})

with open('trained_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
    

@app.route('/')
def home():
    return render_template('index.html')

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # For numerical stability
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = ReLU(Z2)
    Z3 = W3 @ A2 + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def get_predictions(A3):
    return np.argmax(A3, axis=0)

def make_predictions(X, model):
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']
    W3, b3 = model['W3'], model['b3']
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions

def preprocess_image(image_bytes):
    # Open the image
    img = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to grayscale

    # Resize the image to 28x28 pixels
    img = img.resize((28, 28))

    # Convert image to numpy array
    img_array = np.array(img)

    # Invert colors if necessary
    img_array = 255 - img_array  # Invert colors if your model expects white digits on black background

    # Normalize pixel values to [0, 1]
    img_array = img_array / 255.0

    # Flatten the array to shape (784, 1)
    img_array = img_array.reshape(-1, 1)

    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        img_data = data['image']

        # Decode the base64 image
        img_str = base64.b64decode(img_data.split(',')[1])
        img_bytes = img_str

        # Preprocess the image
        img_array = preprocess_image(img_bytes)  # This will have shape (784, 1)

        # Predict
        prediction = make_predictions(img_array, loaded_model)
        digit = int(prediction[0])

        # Return the prediction as JSON
        return jsonify({'digit': digit})

    except Exception as e:
        print('Error:', str(e))
        return jsonify({'error': str(e)}), 500


    

if __name__ == "__main__":
    app.run()