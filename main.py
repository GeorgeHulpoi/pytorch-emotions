import json
import pickle
import torch
import settings 
import redisstream
import emotionscnn

# file = open('keys.pickle', 'rb')
# keys = pickle.load(file)
# file.close()

file = open('data.pickle', 'rb')
data = pickle.load(file)
file.close()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cnn = emotionscnn.EmotionsCNN().to(device)
#stream = redisstream.RedisStream()

nrEpochs = 10
for epoch in range(nrEpochs):
    for i in range(len(data) - 100):
        _data = data[i]
        distribution = json.loads(_data[1].decode())
        image = _data[0]

        cnn.train(image, distribution)
        #print(f'Key {keys[i]} trained.')

    print(f'Epoch {epoch+1} done.')

sum = 0
right = 0
c = 0

for i in range(len(data) - 100, len(data)):
    _data = data[i]
    distribution = json.loads(_data[1].decode())
    image = _data[0]

    distributionTensor = cnn.distributionToTensor(distribution)
    imageTensor = cnn.imageToTensor(image)

    cnn.optimizer.zero_grad()
    predicted = cnn.forward(imageTensor)
    loss = cnn.lossFunc(predicted, distributionTensor)
    sum += loss

    if torch.argmax(distributionTensor) == torch.argmax(predicted):
        right += 1

    c += 1

print(f'Average loss: {sum/c}')
print(f'Right answers: {right}/{c}')