import os
import cv2
import pickle

files = os.listdir('images')

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
    0, # 38
    2, # 39
    0, # 40
    2, # 41
    2, # 42
    0, # 43
    0, # 44
    0, # 45
    0, # 46
    1, # 47
    1, # 48
    2, # 49
    2, # 50
    0, # 51
    0, # 52
    0, # 53
    2, # 54
    2, # 55
    2, # 56
    2, # 57
    2, # 58
    2, # 59
    2, # 60
]
imgs = []

i = 0

angry = 0
happy = 0
surprise = 0

for label in labels:
    if label == 0:
        angry = angry + 1
    elif label == 1:
        happy = happy + 1
    elif label == 2:
        surprise = surprise + 1

for file in files:
    img = cv2.imread(f'images/{file}')
    imgs.append(img)
    i += 1

pickle.dump(list(zip(imgs, labels)), open('real_data.pickle', 'wb'))
    
print(f'Angry: {angry}, Happy: {happy}, Surprise: {surprise}')