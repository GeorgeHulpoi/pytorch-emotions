import cv2
import numpy as np
from pipes import PipeTransform

class CV2ResizePipe(PipeTransform):
    def transform(self, input: np.array) -> np.array:
        return cv2.resize(input, (200, 250))