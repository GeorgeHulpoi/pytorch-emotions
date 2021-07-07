import cv2
import numpy as np
from pipes import PipeTransform

class CV2CropFacePipe(PipeTransform):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def transform(self, input: np.array) -> np.array:
        image = input.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None
        elif len(faces) > 1:
            print(f'Multiple faces ({len(faces)}) detected...')
            return None

        H, W, C = image.shape 
        (x, y, w, h) = faces[0]
        aspect = w/h

        if aspect != 4/5:
            image = self.adjustRatioSize(image, H, x, y, w, h)
        else:
            image = image[y:y+h, x:x+w]

        return image

    def adjustRatioSize(self, image, imageHeight, faceX, faceY, faceWidth, faceHeight):
        # The values with _ at the end represent the new values.
        h_ = np.round((5/4) * faceWidth).astype(int) # cat trebuie sa fie inaltimea
        y_ = np.round(faceY - (h_ - faceHeight) / 2).astype(int) # se ajusteaza noul y incat sa centreze imaginea
        if y_ < 0: # daca iese in sus, se duce inapoi jos
            y_ = 0
        
        if y_ + h_ > imageHeight: # iese in jos
            offset = y_ + h_ - imageHeight # calculeaza cu cat iese
            y_ -= offset # impinge in sus offset-ul

            if y_ < 0: # in cazul asta.. de fapt inaltimea nu incape in imagine
                h_ += y_ 
                y_ = 0

        return image[y_:y_+h_, faceX:faceX+faceWidth]