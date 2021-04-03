import cv2
import numpy as np

from pipes import Pipe

class CV2ResizePipe(Pipe):
    def process(self, input: np.array) -> np.array:
        return cv2.resize(input, (200, 250))