import gc
import sys
import time
import argparse
import queue
import torch
import traceback
import settings
import redisstream
from pipes import Base64ToImagePipe, ImageToCV2Pipe, JsonParserPipe, DistributionToLabelPipe, CV2CropFacePipe, CV2ResizePipe, CV2ToTensorPipe
from pipeline import Pipeline
from model import Model, modelPath, device

parser = argparse.ArgumentParser(description='Test network model from the Redis stream')
parser.add_argument('--size', type=int, help='The number of images for training. It will end the execution (and save the model) after it.')
args = parser.parse_args()

stream = redisstream.RedisStream()
stream.startListening()

imagePipeline = Pipeline().addHandler(Base64ToImagePipe) \
                          .addHandler(ImageToCV2Pipe) \
                          .addHandler(CV2CropFacePipe) \
                          .addHandler(CV2ResizePipe) \
                          .addHandler(CV2ToTensorPipe)          

labelPipeline = Pipeline().addHandler(JsonParserPipe) \
                          .addHandler(DistributionToLabelPipe)


def main():
    counter = 0
    total = 0

    print('Start listening...')
    while True:
        # Extrage din coada. Daca coada este goala, ramane blocat pana cand apare un element
        try:
            key = stream.queue.get_nowait()
            data = stream.getKeyData(key)
            #print(f'Received {key}.')

            image = imagePipeline.execute(data[0])
            label = labelPipeline.execute(data[1].decode())

            if image is None:
                del label
                gc.collect()
                stream.deleteKeyData(key)
                continue

            if Model.test(image, label):
                counter += 1
            total += 1

            del image 
            del label
            gc.collect()
            stream.deleteKeyData(key)

            if args.size is not None:
                if counter == args.size:
                    print(f'Total: {total}')
                    print(f'Correct images: {counter}')
                    print(f'Accuracy: {(counter/total) * 100}%')
                    raise Exception('Size reached')
                elif ((counter / args.size) * 100) % 10 == 0:
                    print(f'{((counter / args.size) * 100)}% done.')

        except queue.Empty:
            time.sleep(1)

if __name__ == '__main__':
    try:
        main()
    except:
        print(traceback.format_exc())
        print('Exit')
        stream.stopListening()