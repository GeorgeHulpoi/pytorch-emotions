import cv2
import numpy as np
import PIL.Image

from pipes import Pipe

class CV2CropFacePipe(Pipe):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def process(self, input: np.array) -> np.array:
        image = input.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return None

        H, W, channels = image.shape 
        (x, y, w, h) = faces[0]
        aspect = w/h

        if aspect != 4/5:
            h_ = np.round((5/4) * w).astype(int)
            y_ = np.round(y - (h_ - h) / 2).astype(int)
            if y_ < 0:
                y_ = 0
            
            if y_ + h_ > H:
                offset = y_ + h_ - H 
                y_ -= offset 

                if y_ < 0:
                    h_ += y_ 
                    y_ = 0

            image = image[y_:y_+h_, x:x+w]
        else:
            image = image[y:y+h, x:x+w]

        return image