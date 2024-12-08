// script.js

// Get the canvas element and its 2D drawing context
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Set initial drawing state
let drawing = false;

// Set canvas background to white
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Add event listeners for mouse and touch events
// (Same as before)

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);
canvas.addEventListener('mousemove', draw);

canvas.addEventListener('touchstart', startDrawingTouch, false);
canvas.addEventListener('touchend', stopDrawingTouch, false);
canvas.addEventListener('touchcancel', stopDrawingTouch, false);
canvas.addEventListener('touchmove', drawTouch, false);

// Drawing functions remain mostly the same

function startDrawing(e) {
  drawing = true;
  draw(e);
}

function stopDrawing() {
  drawing = false;
  ctx.beginPath();
}

function draw(e) {
  if (!drawing) return;
  
  e.preventDefault();

  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  ctx.lineWidth = 15;
  ctx.lineCap = 'square';
  ctx.strokeStyle = 'black';

  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function startDrawingTouch(e) {
  drawing = true;
  drawTouch(e);
}

function stopDrawingTouch() {
  drawing = false;
  ctx.beginPath();
}

function drawTouch(e) {
  if (!drawing) return;
  
  e.preventDefault();

  const rect = canvas.getBoundingClientRect();
  const touch = e.touches[0];
  const x = touch.clientX - rect.left;
  const y = touch.clientY - rect.top;

  ctx.lineWidth = 15;
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'black';

  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
}

// Function to clear the canvas
function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Function to predict the drawn digit
function predictDigit() {
  // Create an off-screen canvas for resizing
  const offscreenCanvas = document.createElement('canvas');
  offscreenCanvas.width = 28;
  offscreenCanvas.height = 28;
  const offscreenCtx = offscreenCanvas.getContext('2d');

  // Draw the current canvas onto the off-screen canvas, resizing it to 28x28
  offscreenCtx.drawImage(canvas, 0, 0, 28, 28);

  // Get the image data from the off-screen canvas
  const dataURL = offscreenCanvas.toDataURL('image/png');

  // Send the image data to the server via POST request
  //change server link once AWS API Gateway and lambda are implemented
  fetch('http://127.0.0.1:5000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ image: dataURL })
  })
  .then(response => response.json())
  .then(data => {
    document.getElementById('result').innerText = 'Prediction: ' + data.digit;
  })
  .catch(error => {
    console.error('Error:', error);
    document.getElementById('result').innerText = 'Error: ' + error.message;
  });
}
