import cv2
import numpy as np
import PIL.Image
import torch

from pipes import Pipe

class CV2ToImagePipe(Pipe):
    def process(self, input: np.array) -> PIL.Image:
        output = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        output = PIL.Image.fromarray(output)
        return output