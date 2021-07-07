import cv2
import numpy as np
import torch
from pipes import PipeTransform

class CV2ToTensorPipe(PipeTransform):
    def transform(self, input: np.array) -> torch.tensor:
        output = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        output = np.moveaxis(output, -1, 0)
        output = torch.from_numpy(output)
        output = output.type(torch.FloatTensor)
        output = output.unsqueeze(0)
        return output