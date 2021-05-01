import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import functools
import operator


modelPath = 'model.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class _Model(nn.Module):
    
    def __init__(self):
        super(_Model, self).__init__()

        imageWidth = int(os.getenv('IMAGE_WIDTH'))
        imageHeight = int(os.getenv('IMAGE_HEIGHT'))
        imageChannels = int(os.getenv('IMAGE_CHANNELS'))
        emotions = int(os.getenv('EMOTIONS'))
        learnRate = float(os.getenv('LEARN_RATE'))

        self.layer1 = nn.Sequential(
            nn.Conv2d(imageChannels, 8, 3),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU())

        fcInputSize = torch.rand(1, imageChannels, imageHeight, imageWidth)
        fcInputSize = self.layer1(fcInputSize)
        fcInputSize = self.layer2(fcInputSize)
        fcInputSize = self.layer3(fcInputSize)
        fcInputSize = self.layer4(fcInputSize)
        fcInputSize = functools.reduce(operator.mul, list(fcInputSize.shape))

        self.fcLayer1 = nn.Linear(fcInputSize, 128)
        self.fcLayer2 = nn.Linear(128, 64)
        self.fcLayer3 = nn.Linear(64, emotions)

        self.lossFunc = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), learnRate)

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        output = output.view([1, -1])
        output = F.relu(self.fcLayer1(output))
        output = F.relu(self.fcLayer2(output))
        output = self.fcLayer3(output)
        return output

    def train(self, image, label):
        image = image.to(device)
        label = label.to(device)

        self.optimizer.zero_grad()
        predicted = self.forward(image)
        #print(predicted)
        #print(label)
        loss = self.lossFunc(predicted, label)
        loss.backward()
        self.optimizer.step()

    def test(self, image, label):
        image = image.to(device)
        label = label.to(device)

        predicted = self.forward(image)
        print(predicted)
        print(label)
        return label[0] == torch.argmax(predicted)

if os.path.exists(modelPath):
    Model = _Model()
    Model.load_state_dict(torch.load(modelPath))
    Model.to(device)
else:
    Model = _Model().to(device)