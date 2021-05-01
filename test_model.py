import os
import re
import cv2
import torch
import numpy as np
import settings
from pipes import CV2CropFacePipe, CV2ResizePipe, CV2ToTensorPipe
from pipeline import Pipeline
from model import Model

files = os.listdir('test')

def sort_func(text):
    return int(text.split('.')[0])

files = sorted(files, key=sort_func)

labels = [
    0, # 1
    0, # 2
    0, # 3
    1, # 4
    1, # 5
    2, # 6
    0, # 7
    2, # 8
    1, # 9
    1, # 10
    1, # 11
    1, # 12
    1, # 13
    1, # 14
    2, # 15
    0, # 16
    1, # 17
    0, # 18
    0, # 19
    0, # 20
    1, # 21
    1, # 22
    2, # 23
    1, # 24
    0, # 25
    1, # 26
    0, # 27
    2, # 28
    2, # 29
    1, # 30
    2, # 31
    1, # 32
    2, # 33
    0, # 34
    1, # 35
    1, # 36
    1, # 37
]

imagePipeline = Pipeline().addHandler(CV2CropFacePipe) \
                          .addHandler(CV2ResizePipe) \
                          .addHandler(CV2ToTensorPipe)
i = 0
counter = 0
total = 0
for file in files:
    img = cv2.imread(f'test/{file}')
    img = imagePipeline.execute(img)
    label = torch.tensor(labels[i]).unsqueeze(0)
    i += 1

    if img is None:
        print(file)
        continue


    if Model.test(img, label):
        counter += 1
    else:
        print(f'{file} failed')
    total += 1

    # cv2.imshow(file, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

print(f'Accuracy: {(counter/total) * 100}%')