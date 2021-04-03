import cv2
import numpy as np
import PIL.Image
import torch

from pipes import Pipe

class CV2ToTensorPipe(Pipe):
    def process(self, input: np.array) -> torch.tensor:
        output = input.copy()
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output = np.moveaxis(output, -1, 0)
        output = torch.from_numpy(output)
        output = output.type(torch.FloatTensor)
        output = output.unsqueeze(0)
        return output