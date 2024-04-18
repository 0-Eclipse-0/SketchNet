import os

import cv2
import torch
import torchvision
import glob
import random
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

from Network import Network
from Dataset import SketchDataset

TAG = "[SketchNet]"

# Data Params
DATASET = "./data"
DATA_SPLIT = (.70, .15, .15) # 75% train, 15% validation, 15% test

# Training Params
LEARNING_RATE = 0.002
MOMENTUM = 0.9
EPOCHS = 3

# Transforms
transform = A.Compose(
    [
        A.Resize(height=256, width=256, interpolation=cv2.INTER_LANCZOS4),
        ToTensorV2()
    ]
)

# Prepare runtime for training
def prepRuntime():
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        print(TAG, "CUDA available, using GPU")
    else:
        torch.set_default_device("cpu")
        print(TAG, "CUDA isn't available, using CPU")

# Convert data
def prepareDataset():
    test_data, train_data, validation_data = [], [], []
    # Get class names
    index_map = {}
    class_map = {}

    # Map: Index <-> Class
    for k, v in enumerate(os.listdir(DATASET)):
        index_map[k], class_map[v] = v, k

    # Get all paths
    with open(DATASET + '/filelist.txt', 'r') as f:
        files = [DATASET + '/' + line.strip() for line in f.readlines()]
        length = len(files)

        random.shuffle(files)

        # Fold dataset
        train_data = files[:int(length * DATA_SPLIT[0])]
        validation_data = files[int(length * DATA_SPLIT[0]):int(length * sum(DATA_SPLIT[:2]))]
        test_data = files[int(length * sum(DATA_SPLIT[:2])):]

    # Output statistics
    print(TAG, "Dataset Sizes:")
    print("\t- Train: %s\n\t- Test: %s\n\t- Validation: %s" % (len(train_data), len(test_data), len(validation_data)))

    return [index_map, class_map], [train_data, validation_data, test_data]

# Train Network
def train(train_loader, model, criterion, optimizer):
    for epoch in range(EPOCHS):
        running_loss = 0.0

        for iter, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward() # Backwards pass
            optimizer.step() # Update weights

            running_loss += loss

            if iter % 1000 == 999:
                print(f"\t[Training] Epoch: {epoch}, Iter: {iter} / {len(train_loader)}, Loss: {running_loss}")
                
        torch.save(model.state_dict(), f'./epoch_{epoch}.pth')

if __name__ == '__main__':
    # Preparation
    prepRuntime()
    classes, data = prepareDataset()

    # Load datasets
    train_dataset = SketchDataset(data[0], classes, transform)
    validation_dataset = SketchDataset(data[1], classes, transform)
    test_dataset = SketchDataset(data[2], classes, transform)

    # Batching
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    #for x, y in train_dataloader:
    #	print(x.dtype, y) 
    	
    # Training
    net = Network()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    train(train_dataloader, net, loss, optimizer)
