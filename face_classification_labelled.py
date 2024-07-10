from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import cv2
from PIL import Image

# Define transformations for the training and testing data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

expression_map = {
            'neutral': 0,
            'happy': 1,
            'sad': 2,
            'angry': 3,
            'surprise': 4,
            'fear': 5,
            'disgust': 6
        }
# Custom Dataset class to load images
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.pgm'):
                    self.image_files.append(os.path.join(root, file))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (32, 32))

        # Extract labels from filename
        img_name = os.path.basename(img_path)
        parts = img_name.split('_')
    
        is_mitchell = int(parts[0]=='mitchell')  # Binary label: 0 or 1
        # Map expression to an integer label
        expression = expression_map[parts[2]]
        if self.transform:
            image = self.transform(image)
        return image, is_mitchell, expression

def img_loader(filename):
    return cv2.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE), (32,32))


# Define the Multi-Task Learning model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Shared fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        
        # Separate heads for each task
        self.fc3_mitchell = nn.Linear(84, 1)  # Binary classification: Mitchell or not
        self.fc3_expression = nn.Linear(84, 7)  # Multi-class classification: facial expressions
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Task 1: Is this image a picture of Mitchell?
        is_mitchell = torch.sigmoid(self.fc3_mitchell(x))
        
        # Task 2: What's the facial expression in the picture?
        expression = self.fc3_expression(x)
        
        return is_mitchell, expression
    
def train(check_point_file, data_dir, *args, **kwargs):
    """
        Training the model
    """

    dataset = FaceDataset(root_dir=data_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Instantiate the model
    model = Net()

    # Loss functions and optimizer
    criterion_mitchell = nn.BCELoss()  # Binary Cross Entropy Loss for Task 1
    criterion_expression = nn.CrossEntropyLoss()  # Cross Entropy Loss for Task 2
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    num_epochs = 200
    image_count = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss_mitchell = 0.0
        running_loss_expression = 0.0
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels_mitchell, labels_expression = data
            optimizer.zero_grad()
            
            outputs_mitchell, outputs_expression = model(inputs)
            loss_mitchell = criterion_mitchell(outputs_mitchell.squeeze(), labels_mitchell.float())
            loss_expression = criterion_expression(outputs_expression, labels_expression)
            loss = loss_mitchell + loss_expression
            loss.backward()
            optimizer.step()
            
            running_loss_mitchell += loss_mitchell.item()
            running_loss_expression += loss_expression.item()
            

        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                f'Loss Mitchell: {running_loss_mitchell / 100:.4f}, '
                f'Loss Expression: {running_loss_expression / 100:.4f}')
                

    print('Finished Training')
    torch.save(model, check_point_file)
    print('Saved the model')

def test(check_point_file, test_dir, *args, **kwargs):
    # Evaluation
    model = torch.load(check_point_file)

    model.eval()
    correct_mitchell = 0
    total_mitchell = 0
    correct_expression = 0
    total_expression = 0

    dataset = FaceDataset(root_dir=test_dir, transform=transform)
    test_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    with torch.no_grad():
        for data in test_loader:
            images, labels_mitchell, labels_expression = data
            outputs_mitchell, outputs_expression = model(images)
            
            # Evaluate Task 1
            predicted_mitchell = (outputs_mitchell.squeeze() > 0.5).int()
            total_mitchell += labels_mitchell.size(0)
            correct_mitchell += (predicted_mitchell == labels_mitchell).sum().item()
            
            # Evaluate Task 2
            _, predicted_expression = torch.max(outputs_expression, 1)
            total_expression += labels_expression.size(0)
            correct_expression += (predicted_expression == labels_expression).sum().item()

    print(f'Accuracy of the network on Task 1 (Mitchell): {100 * correct_mitchell / total_mitchell:.2f}%')
    print(f'Accuracy of the network on Task 2 (Expression): {100 * correct_expression / total_expression:.2f}%')


def predict(check_point_file, image_path):
    # Load the trained model
    model = torch.load(check_point_file)
    model.eval()

    # Transform the input image
    #image = Image.open(image_path)
    #image = transform(image)  # Add batch dimension
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (32, 32))
    image = transform(image)

    with torch.no_grad():
        is_mitchell, expression = model(image)
        is_mitchell = (is_mitchell.squeeze() > 0.5).int().item()
        expression = torch.argmax(expression, dim=1).item()

    expression_map = {
        0: 'neutral',
        1: 'happy',
        2: 'sad',
        3: 'angry',
        4: 'surprise',
        5: 'fear',
        6: 'disgust'
    }

    print(f"Is this image a picture of Mitchell? {'Yes' if is_mitchell == 1 else 'No'}")
    print(f"Facial expression: {expression_map[expression]}")

def main():
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./faces', metavar='N',
                        help='Path to directory containing faces dataset.')
    parser.add_argument('--predict', type=str, metavar='N',
                        help='Path to image for prediction.')
    
    args = parser.parse_args()
    data_dir = args.data
    checkpoint_dir = './check_point/fc_model.pt'

    if args.predict:
        print("Run the prediction")
        predict(checkpoint_dir, args.predict)
    else:
        print("Run model training and testing")
        train(checkpoint_dir, data_dir)
        test(checkpoint_dir, data_dir + '/test')


if __name__ == '__main__':
    main()