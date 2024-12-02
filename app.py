from flask import Flask, request, jsonify, render_template
import os
import cv2
import torch
import numpy as np
import pandas as pd
from torchvision import transforms, models
from PIL import Image

# Parameters
MODEL_PATH = 'resnet_finetuned.pth'
VAL_CSV = 'data/train/_classes.csv'  # Path to the CSV file with class names
NUM_CLASSES = 4  # Update as per your model's classes

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load species names from CSV
def load_species_names(csv_path):
    df = pd.read_csv(csv_path)
    return df.columns[1:].tolist()  # Assuming the first column is image ID and rest are species

species_names = load_species_names(VAL_CSV)

# Load model
def load_model(model_path, num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model(MODEL_PATH, NUM_CLASSES)

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def segment_trees(image):
    """ Segment trees using watershed algorithm. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    unknown = cv2.subtract(dilated, np.uint8(sure_fg))
    _, markers = cv2.connectedComponents(np.uint8(sure_fg))
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    return np.unique(markers).shape[0] - 1  # Count excluding background

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Load image and process
    img = Image.open(filepath).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        output = model(img_tensor)
        preds = torch.sigmoid(output).squeeze().cpu().numpy()

    # Map predictions to species names
    predicted_species = [
        species_names[i] for i, prob in enumerate(preds) if prob > 0.5
    ]

    # Segment trees
    cv_image = cv2.imread(filepath)
    tree_count = segment_trees(cv_image)

    os.remove(filepath)  # Clean up uploaded file
    return jsonify({
        'predictions': predicted_species,
        'tree_count': tree_count
    })

if __name__ == "__main__":
    app.run(debug=True)
    
    

    
