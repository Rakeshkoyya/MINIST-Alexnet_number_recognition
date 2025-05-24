# MNIST Number Recognition using AlexNet

A web-based application that recognizes handwritten digits (0-9) using an AlexNet neural network model trained on the MNIST dataset.

## Project Overview

This project implements a digit recognition system that:
- Uses AlexNet neural network architecture
- Trains on the MNIST dataset
- Provides a web interface for drawing and predicting digits
- Includes a confidence meter for prediction results

## Project Structure

```
MINIST-Alexnet/
├── alexnet_model.py     # AlexNet model implementation
├── app.py               # Flask backend server
├── trainer.py           # Training script
├── trained_model_k13.pth # Pre-trained model weights
├── data/               # MNIST dataset
├── frontend/           # Vue.js frontend
└── README.md           # This file
```

## Features

- Interactive drawing canvas for input
- Real-time digit prediction
- Confidence score display
- Clear canvas functionality
- Web-based interface using Vue.js
- Backend API using Flask

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install torch flask numpy
   ```

3. Start the backend server:
   ```bash
   python app.py
   ```

4. Navigate to the frontend directory and start the development server:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Usage

1. Draw a digit (0-9) on the canvas
2. Click "Predict" to get the model's prediction
3. View the predicted digit and confidence score
4. Use the "Clear" button to start over

## Model Architecture

The project uses AlexNet architecture adapted for MNIST digit recognition:
- Input size: 28x28 grayscale images
- Modified AlexNet layers for smaller input size
- Output layer with 10 units (one for each digit)
- Softmax activation for probability distribution

## Training

The model was trained using the MNIST dataset and saved as `trained_model_k13.pth`. The training script can be found in `trainer.py`.

## Contributing

Feel free to contribute to this project by:
- Improving the model architecture
- Enhancing the web interface
- Adding new features
- Optimizing performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.
