import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, models
from src.dataset import Dataset  # Ensure Dataset class is in src/dataset.py

# Parameters
IMG_DIR = 'data/train'  # Directory containing images
MODEL_PATH = 'resnet_finetuned.pth'
VAL_IMG = 'data/train'
VAL_CSV = 'data/train/_classes.csv'
BATCH_SIZE = 16
NUM_CLASSES = 4  # Set to the number of classes in your dataset

def preprocess_image(image):
    """ Convert to grayscale and apply Gaussian blur. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def segment_trees(image):
    """ Segment the trees in the image using watershed algorithm. """
    # Apply preprocessing
    blurred = preprocess_image(image)

    # Apply thresholding to get a binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find sure foreground area
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Find unknown region
    unknown = cv2.subtract(dilated, np.uint8(sure_fg))

    # Marker labelling
    _, markers = cv2.connectedComponents(np.uint8(sure_fg))

    # Add one to all labels to distinguish sure regions from unknown
    markers = markers + 1
    markers[unknown == 255] = 0  # Mark unknown region as 0

    # Apply watershed algorithm
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  # Mark boundaries in red

    return markers, image

def load_model(num_classes, model_path):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    return model

def evaluate(model, val_loader, device):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5  # Threshold for multi-label classification
            predictions.append(preds.cpu())
            actuals.append(labels.cpu())
    return torch.cat(predictions), torch.cat(actuals)

def visualize(predictions, actuals, val_loader, class_names, tree_counts):
    inv_transform = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    
    for i, (images, _) in enumerate(val_loader):
        images = images[:4]  # Show the first 4 images in the batch
        predictions_sample = predictions[i * BATCH_SIZE: i * BATCH_SIZE + 4]
        actuals_sample = actuals[i * BATCH_SIZE: i * BATCH_SIZE + 4]
        
        plt.figure(figsize=(10, 10))
        for j in range(images.size(0)):
            img = inv_transform(images[j]).permute(1, 2, 0).numpy()
            pred_labels = [class_names[k] for k, pred in enumerate(predictions_sample[j]) if pred]
            true_labels = [class_names[k] for k, act in enumerate(actuals_sample[j]) if act]
            tree_count = tree_counts[i * BATCH_SIZE + j]  # Get the tree count for this image
            
            plt.subplot(2, 2, j + 1)
            plt.imshow(img)
            plt.title(f"Predicted: {', '.join(pred_labels)}\nActual: {', '.join(true_labels)}") #\nEstimated Trees: {tree_count}
            plt.axis('off')
        
        plt.show()
        break  # Only display one batch for visualization

def main():
    # Load the validation dataset
    val_df = pd.read_csv(VAL_CSV)
    class_names = val_df.columns[1:].tolist()  # Adjust as per your CSV structure
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = Dataset(dataframe=val_df, img_dir=VAL_IMG, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(NUM_CLASSES, MODEL_PATH).to(device)

    # Evaluate model
    predictions, actuals = evaluate(model, val_loader, device)
    
    # Segment trees and count them
    tree_counts = []
    for img_name in os.listdir(IMG_DIR):
        img_path = os.path.join(IMG_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Segment trees in the image
        markers, segmented_img = segment_trees(img)

        # Count trees
        tree_count = np.unique(markers).shape[0] - 1  # Exclude the background
        tree_counts.append(tree_count)

    # Visualize some results
    visualize(predictions, actuals, val_loader, class_names, tree_counts)

if __name__ == "__main__":
    main()
