import cv2
import numpy as np
import PIL.Image
from pipes import PipeTransform

class ImageToCV2Pipe(PipeTransform):
    def transform(self, input: PIL.Image) -> np.array:
        output = np.array(input)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output