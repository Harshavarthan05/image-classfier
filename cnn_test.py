import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import os

print("Current working directory:", os.getcwd())

print("Does 'dataset' exist?", os.path.exists('dataset'))
print("Does 'dataset/train' exist?", os.path.exists('dataset/train'))
print("Does 'dataset/test' exist?", os.path.exists('dataset/test'))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128,128)),
    transforms.Normalize((0.5,),(0.5,))
])

train_data = datasets.ImageFolder('dataset/train', transform = transform)
test_data = datasets.ImageFolder('dataset/test', transform = transform)

train_loader = DataLoader(train_data, shuffle = True, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            ) 

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*32*32,512),
            nn.ReLU(),
            nn.Linear(512,2)
        )  

    def forward(self,x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.01)

epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        y_pred = model(images)
        loss = loss_fn(y_pred, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"EPoch: {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.3f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _,predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}")

def predict_image(img_path):
    image = Image.open(img_path)
    image = transform(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _,predicted = torch.max(output, 1)
        classes = ['cat','dog']
        return classes[predicted.item()]

print(predict_image('dataset/test/dogs/dog_test1.jpg'))