import cv2
import numpy as np
import PIL.Image

from pipes import Pipe

class ImageToCV2Pipe(Pipe):
    def process(self, input: PIL.Image) -> PIL.Image:
        output = np.array(input)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output