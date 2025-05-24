from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
import base64
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
from alexnet_model import AlexNet  # Import your model definition
import matplotlib.pyplot as plt
import re


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model
model = AlexNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU") 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("trained_model_k13.pth", map_location=device))
model.to(device)
model.eval()

# Image transform (for 28x28 MNIST)
transform = transforms.Compose([
    transforms.Grayscale(),  # Ensure it's 1-channel
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # data = request.json['image']
        # image_data = base64.b64decode(data.split(',')[1])
        # image = Image.open(BytesIO(image_data)).convert("L")  # Convert to grayscale

        image_data = request.json['image']
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        image = Image.open(BytesIO(base64.b64decode(image_data))).convert('L')  # Convert to grayscale

        # Preprocess
        image = ImageOps.invert(image)
        
        # Resize to 28x28 like MNIST
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # Apply same transform used during training
        transform = transforms.Compose([
            transforms.ToTensor(),                      # Converts to [0,1]
            transforms.Normalize((0.5,), (1.0,))        # Match training normalization
        ])
        image = transform(image).unsqueeze(0).to(device)

        # plt.imsave("debug_input.png", image[0].cpu().numpy().squeeze() )

        # Predict
        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probabilities[0][prediction].item()
        return jsonify({
            'prediction': prediction,
            'confidence': f"{confidence:.2%}"  # Format as percentage with 2 decimal places
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
