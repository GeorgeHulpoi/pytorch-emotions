import gc
import time
import argparse
import queue
import traceback
import redisstream
from pipes import Pipeline, Base64ToImagePipe, ImageToCV2Pipe, JsonParserPipe, DistributionToLabelPipe, CV2CropFacePipe, CV2ResizePipe, CV2ToTensorPipe
from model import Model

parser = argparse.ArgumentParser(description='Test network model from the Redis stream')
parser.add_argument('--size', type=int, help='The number of images for testing. It will end the execution after it.')
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
    total = 0

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
        stream.stopListening()