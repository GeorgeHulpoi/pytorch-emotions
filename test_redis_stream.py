import gc
import sys
import time
import queue
import torch
import traceback
import settings
import redisstream
from pipes import Base64ToImagePipe, ImageToCV2Pipe, JsonParserPipe, DistributionToLabelPipe, CV2CropFacePipe, CV2ResizePipe, CV2ToTensorPipe
from pipeline import Pipeline
from model import Model, modelPath, device

stream = redisstream.RedisStream()
stream.startListening()

imagePipeline = Pipeline().addHandler(Base64ToImagePipe) \
                          .addHandler(ImageToCV2Pipe) \
                          .addHandler(CV2CropFacePipe) \
                          .addHandler(CV2ResizePipe) \
                          .addHandler(CV2ToTensorPipe)          

labelPipeline = Pipeline().addHandler(JsonParserPipe) \
                          .addHandler(DistributionToLabelPipe)

counter = 0
total = 0

def main():
    global counter, total
    print('Start listening...')
    while True:
        # Extrage din coada. Daca coada este goala, ramane blocat pana cand apare un element
        try:
            key = stream.queue.get_nowait()
            data = stream.getKeyData(key)
            print(f'Received {key}.')

            image = imagePipeline.execute(data[0])
            label = labelPipeline.execute(data[1].decode())

            if Model.test(image, label):
                counter += 1
            total += 1

            del image 
            del label
            gc.collect()
            stream.deleteKeyData(key)

        except queue.Empty:
            time.sleep(1)

if __name__ == '__main__':
    try:
        main()
    except:
        #print(traceback.format_exc())
        #print('Exit')
        stream.stopListening()

        print(f'Total: {total}')
        print(f'Correct images: {counter}')
        print(f'Accuracy: {(counter/total) * 100}%')