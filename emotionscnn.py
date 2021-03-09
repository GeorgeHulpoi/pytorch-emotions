import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from converter import Converter

if __name__ == "__main__":
    sys.exit('Don\'t run as main script!')

class EmotionsCNN(nn.Module):
    
    def __init__(self):
        super(EmotionsCNN, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        imageWidth = int(os.getenv('IMAGE_WIDTH'))
        imageHeight = int(os.getenv('IMAGE_HEIGHT'))
        filters = int(os.getenv('FILTERS'))
        filterSize = int(os.getenv('FILTER_SIZE'))
        poolSize = int(os.getenv('POOL_SIZE'))
        emotions = int(os.getenv('EMOTIONS'))
        learnRate = float(os.getenv('LEARN_RATE'))

        # Fully Connected Input Size
        fcInputSize = (imageWidth - 2 * (filterSize // 2)) * (imageHeight - 2* (filterSize // 2)) * filters // (2 * poolSize)

        self.convLayer = nn.Conv2d(3, filters, filterSize)
        # testeaza max, sum si average
        self.poolLayer = nn.MaxPool2d(poolSize)

        self.fcLayer = nn.Linear(fcInputSize, emotions)

        self.lossFunc = nn.L1Loss()
        self.optimizer = torch.optim.SGD(self.parameters(), learnRate)

    def forward(self, input):
        output = self.convLayer(input)
        output = self.poolLayer(output)
        output = F.relu(output)
        output = output.view([1, -1])
        output = self.fcLayer(output)
        return output

    def distributionToTensor(self, distribution):
        vector = [float(distribution['angry']),  float(distribution['happy']), float(distribution['surprise'])]
        target = torch.FloatTensor(vector)
        target = target.unsqueeze(0).to(self.device)
        return target

    def imageToTensor(self, image):
        return Converter.base64ImageToTensor(image).unsqueeze(0).to(self.device)

    def train(self, image, distribution):
        target = self.distributionToTensor(distribution)
        imageTensor = self.imageToTensor(image)
        self.optimizer.zero_grad()
        predicted = self.forward(imageTensor)
        loss = self.lossFunc(predicted, target)
        loss.backward()
        self.optimizer.step()