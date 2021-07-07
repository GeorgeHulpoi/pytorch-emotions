import os
import torch
import torch.nn as nn
import torch.nn.functional as F

modelPath = 'model.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NeuronalNetwork(nn.Module):
    
    def __init__(self):
        super(NeuronalNetwork, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(8, 16, 3), nn.ReLU(), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer5 = nn.Sequential(nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer6 = nn.Sequential(nn.Conv2d(128, 256, 3), nn.ReLU(), nn.MaxPool2d(2, 2))

        self.fcLayer1 = nn.Linear(256, 512)
        self.fcLayer2 = nn.Linear(512, 256)
        self.fcLayer3 = nn.Linear(256, 256)
        self.fcLayer4 = nn.Linear(256, 3)

        self.lossFunc = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        output = self.layer6(output)

        output = torch.flatten(output, start_dim=1)
        output = F.relu(self.fcLayer1(output))
        output = F.relu(self.fcLayer2(output))
        output = F.relu(self.fcLayer3(output))
        output = self.fcLayer4(output)
        return output

    def train(self, image, label):
        image = image.to(device)
        label = label.to(device)

        self.optimizer.zero_grad()
        predicted = self.forward(image)
        loss = self.lossFunc(predicted, label)
        loss.backward()
        self.optimizer.step()

    def test(self, image, label):
        image = image.to(device)
        label = label.to(device)

        predicted = self.forward(image)
        return label[0] == torch.argmax(predicted)

Model = NeuronalNetwork().to(device)
if os.path.exists(modelPath):
    Model.load_state_dict(torch.load(modelPath))