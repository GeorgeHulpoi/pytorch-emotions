import json
import pickle
import torch
import settings 
import redisstream
import emotionscnn

file = open('keys.pickle', 'rb')
keys = pickle.load(file)
file.close()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cnn = emotionscnn.EmotionsCNN().to(device)
stream = redisstream.RedisStream()

nrEpochs = 10
for epoch in range(nrEpochs):
    for i in range(len(keys) - 100):
        data = stream.getKeyData(keys[i])
        distribution = json.loads(data[1].decode())
        image = data[0]

        cnn.train(image, distribution)
        #print(f'Key {keys[i]} trained.')

    print(f'Epoch {epoch+1} done.')

right = 0
c = 0

for i in range(len(keys) - 100, len(keys)):
    key = keys[i]
    data = stream.getKeyData(key)
    distribution = json.loads(data[1].decode())
    image = data[0]

    distributionTensor = cnn.distributionToTensor(distribution)
    target = torch.argmax(distributionTensor)
    imageTensor = cnn.imageToTensor(image)

    cnn.optimizer.zero_grad()
    predicted = cnn.forward(imageTensor)
    loss = cnn.lossFunc(predicted, target.unsqueeze(0))

    if torch.argmax(distributionTensor) == torch.argmax(predicted):
        right += 1

    c += 1

print(f'Right answers: {right}/{c}')