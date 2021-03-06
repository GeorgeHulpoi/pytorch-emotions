import gc
import os
import time
import queue
import argparse
import torch
import traceback
import redisstream
from pipes import Pipeline, Base64ToImagePipe, ImageToCV2Pipe, JsonParserPipe, DistributionToLabelPipe, CV2CropFacePipe, CV2ResizePipe, CV2ToTensorPipe
from model import Model, modelPath

parser = argparse.ArgumentParser(description='Train network model from the Redis stream')
parser.add_argument('--size', type=int, help='The number of images for training. It will end the execution (and save the model) after it.')
parser.add_argument('--shutdown', action='store', const='NoValue', nargs='?', help='Shutdown the Windows at the end of training.')
args = parser.parse_args()

stream = redisstream.RedisStream()
stream.startListening()

imagePipeline = Pipeline().pipe(Base64ToImagePipe) \
                          .pipe(ImageToCV2Pipe) \
                          .pipe(CV2CropFacePipe) \
                          .pipe(CV2ResizePipe) \
                          .pipe(CV2ToTensorPipe)          

labelPipeline = Pipeline().pipe(JsonParserPipe) \
                          .pipe(DistributionToLabelPipe)

def main():
    counter = 0

    while True:
        try:
            key = stream.queue.get_nowait()
            data = stream.getKeyData(key)

            image = imagePipeline.execute(data[0])
            label = labelPipeline.execute(data[1].decode())

            if image is None:
                del label
                gc.collect()
                stream.deleteKeyData(key)
                continue

            Model.train(image, label)

            del image 
            del label
            gc.collect()
            stream.deleteKeyData(key)

            counter += 1
            if args.size is not None:
                if counter == args.size:
                    if args.shutdown is not None:
                        torch.save(Model.state_dict(), modelPath)
                        stream.stopListening()
                        os.system("shutdown /s /t 1")
                    else:
                        raise Exception('Size reached')
                elif counter != 0 and ((counter / args.size) * 100) % 10 == 0:
                    print(f'{((counter / args.size) * 100)}% done.')

        except queue.Empty:
            time.sleep(1)

if __name__ == '__main__':
    print('Start listening...')
    try:
        main()
    except:
        print(traceback.format_exc())
        print('Exit')
        torch.save(Model.state_dict(), modelPath)
        stream.stopListening()