import cv2
import numpy as np
import PIL.Image
from pipes import PipeTransform

class CV2ToImagePipe(PipeTransform):
    def transform(self, input: np.array) -> PIL.Image:
        output = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        output = PIL.Image.fromarray(output)
        return output