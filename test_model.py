import torch
import pickle
import settings
from pipes import CV2ToImagePipe, Base64ToImagePipe, ImageToCV2Pipe, JsonParserPipe, DistributionToLabelPipe, CV2CropFacePipe, CV2ResizePipe, CV2ToTensorPipe
from pipeline import Pipeline
from model import Model


with open('test.pickle', 'rb') as file:
    test = pickle.load(file)

imagePipeline = Pipeline().addHandler(Base64ToImagePipe) \
                          .addHandler(ImageToCV2Pipe) \
                          .addHandler(CV2CropFacePipe) \
                          .addHandler(CV2ResizePipe) \
                          .addHandler(CV2ToImagePipe)          

labelPipeline = Pipeline().addHandler(JsonParserPipe) \
                          .addHandler(DistributionToLabelPipe)

counter = 0
total = 0

for (image, label) in test:
    label = int(label)

    if label == -1:
        continue 

    image = imagePipeline.execute(image)
    label = torch.tensor(label).unsqueeze(0)

    if image == None:
        continue


    image.show()
    # if Model.test(image, label):
    #     counter += 1
    # total += 1
        

#print(f'Accuracy: {(counter/total) * 100}%')