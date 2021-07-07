import torch
import pickle
from pipes import Pipeline, CV2CropFacePipe, CV2ResizePipe, CV2ToTensorPipe
from model import Model

imagePipeline = Pipeline().pipe(CV2CropFacePipe) \
                          .pipe(CV2ResizePipe) \
                          .pipe(CV2ToTensorPipe)

batch = pickle.load(open('real_data.pickle', 'rb'))

counter = 0
total = 0
for data in batch:
    img = imagePipeline.execute(data[0])
    label = torch.tensor(data[1]).unsqueeze(0)

    if img is None:
        continue

    if Model.test(img, label):
        counter += 1

    total += 1

print(f'Accuracy: {(counter/total) * 100}%')