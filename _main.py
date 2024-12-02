import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import pandas as pd
from src.dataset import Dataset  # Ensure your Dataset class is saved in dataset.py

# Paths and Parameters
TEST_IMG = 'data/species/train'
TEST_CSV = 'data/species/train/_classes.csv'
VAL_IMG = 'data/species/train'
VAL_CSV = 'data/species/train/_classes.csv'
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 4  # Set to the number of main classes in your dataset

# Custom model with two heads
class ResNetMultiHead(nn.Module):
    def __init__(self, num_classes, num_wildfire_classes=1):
        super(ResNetMultiHead, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        
        # Main classification head for NUM_CLASSES
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        
        # Additional binary classification head for wildfire detection
        self.wildfire_head = nn.Linear(self.base_model.fc.in_features, num_wildfire_classes)

    def forward(self, x):
        base_output = self.base_model(x)
        wildfire_output = self.wildfire_head(x)
        return base_output, wildfire_output

def main():
    # Load the dataset
    train_df = pd.read_csv(TEST_CSV)
    val_df = pd.read_csv(VAL_CSV)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets and dataloaders
    train_dataset = Dataset(dataframe=train_df, img_dir=TEST_IMG, transform=transform)
    val_dataset = Dataset(dataframe=val_df, img_dir=VAL_IMG, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model
    model = ResNetMultiHead(num_classes=NUM_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss functions and optimizer
    criterion_main = nn.BCEWithLogitsLoss()  # For multi-class classification
    criterion_wildfire = nn.BCEWithLogitsLoss()  # For binary wildfire classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, (labels_main, labels_wildfire) in train_loader:
            images = images.to(device)
            labels_main = labels_main.to(device).float()
            labels_wildfire = labels_wildfire.to(device).float()

            optimizer.zero_grad()
            outputs_main, outputs_wildfire = model(images)

            # Calculate losses for both heads
            loss_main = criterion_main(outputs_main, labels_main)
            loss_wildfire = criterion_wildfire(outputs_wildfire, labels_wildfire)
            loss = loss_main + loss_wildfire  # Combined loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")

        # Validation loop
        model.eval()
        correct_main = 0
        correct_wildfire = 0
        total_main = 0
        total_wildfire = 0

        with torch.no_grad():
            for images, (labels_main, labels_wildfire) in val_loader:
                images = images.to(device)
                labels_main = labels_main.to(device).float()
                labels_wildfire = labels_wildfire.to(device).float()

                outputs_main, outputs_wildfire = model(images)
                
                # Calculate predictions
                predictions_main = torch.sigmoid(outputs_main) > 0.5
                predictions_wildfire = torch.sigmoid(outputs_wildfire) > 0.5
                
                # Accuracy for main classification
                total_main += labels_main.size(0) * NUM_CLASSES
                correct_main += (predictions_main == labels_main).sum().item()

                # Accuracy for wildfire classification
                total_wildfire += labels_wildfire.size(0)
                correct_wildfire += (predictions_wildfire == labels_wildfire).sum().item()

        accuracy_main = 100 * correct_main / total_main
        accuracy_wildfire = 100 * correct_wildfire / total_wildfire

        print(f'Validation Accuracy - Main Classification: {accuracy_main:.2f}%')
        print(f'Validation Accuracy - Wildfire Classification: {accuracy_wildfire:.2f}%')

    # Save the trained model
    torch.save(model.state_dict(), 'resnet_multiclass_wildfire.pth')
    print("Model saved as resnet_multiclass_wildfire.pth")

if __name__ == "__main__":
    main()
