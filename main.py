import json
import pickle
import torch
import random
import settings 
import redisstream
import emotionscnn

from converter import Converter

with open('keys.pickle', 'rb') as file:
    keys = pickle.load(file)

with open('test.pickle', 'rb') as file:
    test = pickle.load(file)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cnn = emotionscnn.EmotionsCNN().to(device)
stream = redisstream.RedisStream()

nrEpochs = 10
for epoch in range(nrEpochs):
    random.shuffle(keys)

    counter = 0
    correct = 0

    for i in range(len(keys) - 100):
        data = stream.getKeyData(keys[i])
        (image, label) = Converter.dataToTensor(data)

        cnn.train(image.to(device), label.to(device))

    for i in range(len(keys) - 100, len(keys)):
        data = stream.getKeyData(keys[i])
        (image, label) = Converter.dataToTensor(data)

        if cnn.test(image.to(device), label.to(device)):
            correct += 1
        counter += 1

    print(f'Correct: {correct}, Total Tests: {counter}, Accuracy: {(correct/counter) * 100}%')

    counter = 0
    correct = 0

    for i in range(len(test)):
        image = test[i][0]
        label = int(test[i][1])

        if label == -1:
            continue

        image = Converter.base64ImageToTensor(image).to(device)
        label = torch.tensor(label).unsqueeze(0).to(device)

        if cnn.test(image, label):
            correct += 1
        counter += 1

    print(f'Correct: {correct}, Total Tests: {counter}, Accuracy: {(correct/counter) * 100}%')

    print(f'Epoch {epoch+1} done.')