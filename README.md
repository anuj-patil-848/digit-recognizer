# Flask-Based Handwritten Digit Recognizer

This project implements a web application for recognizing handwritten digits using a pre-trained neural network model. The backend is powered by **Flask**, and the model is loaded from a serialized file (`trained_model.pkl`). Users can send digit images to the application via a POST request, and the application returns the recognized digit.

## Features

- Accepts base64-encoded images of handwritten digits via POST requests
- Preprocesses input images (resizing, normalization, etc.) to prepare for model prediction
- Uses a simple neural network for digit recognition
- Outputs predictions as JSON
- CORS support for cross-origin requests

## Requirements

### Python Libraries
- `pickle`
- `numpy`
- `Flask`
- `Flask-CORS`
- `Pillow`

Install the dependencies using:
```bash
pip install numpy flask flask-cors pillow
```

## How It Works

1. **Model Loading**
   - The neural network model is loaded from `trained_model.pkl`, which contains the weights and biases for a three-layer network.

2. **Image Preprocessing**
   - Images are resized to 28x28 pixels and normalized to values between 0 and 1.
   - Images are converted to grayscale and flattened to a vector format suitable for the model.

3. **Forward Propagation**
   - The model uses ReLU activation for hidden layers and softmax activation for the output layer.
   - Predictions are computed by taking the argmax of the output probabilities.

4. **REST API**
   - The `/predict` endpoint accepts base64-encoded images, preprocesses them, and returns the predicted digit as JSON.

## API Endpoints

### `GET /`
Serves the home page (index.html). This can be used to provide a user interface for testing (optional).

### `POST /predict`
- **Description:** Accepts a base64-encoded image of a handwritten digit and returns the recognized digit.
- **Request Body:** JSON object with the following key:
  - `image`: A base64-encoded string representing the image.
- **Response:** JSON object with the following key:
  - `digit`: The predicted digit (0-9).

Example Request:
```json
{
  "image": "data:image/png;base64,iVBORw0KGgo..."
}
```

Example Response:
```json
{
  "digit": 5
}
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/your-username/flask-digit-recognizer.git
cd flask-digit-recognizer
```

2. Run the application:
```bash
python app.py
```

3. Send a POST request to the `/predict` endpoint with a base64-encoded image of a digit.

4. (Optional) Open `index.html` in a browser to test the interface.

## Project Structure

```
flask-digit-recognizer/
├── app.py                # Main Flask application
├── trained_model.pkl     # Pre-trained model file
├── templates/
│   └── index.html        # HTML template (optional)
├── static/
│   └── styles.css        # Optional CSS for the interface
└── README.md             # Project documentation
```

## Pre-trained Model

The `trained_model.pkl` file contains the weights and biases for a simple neural network:
- Input layer: 784 units (28x28 pixels)
- Two hidden layers: ReLU activations
- Output layer: Softmax activation with 10 units (for digits 0-9)

## Contributions

Feel free to contribute to this project by submitting pull requests. Suggestions for improving the model or API are welcome.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The pre-trained model was created using synthetic training data derived from the MNIST dataset.
- Flask and Pillow libraries for their ease of use in web development and image processing.
