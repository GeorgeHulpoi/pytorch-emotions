import gc
import sys
import json
import time
import pickle
import queue
import traceback
import settings
import redisstream
from converter import Converter
from pipes import Base64ToImagePipe, ImageToCV2Pipe, CV2CropFacePipe, CV2ToTensorPipe, ImageToTensorPipe, CV2ToImagePipe, CV2ResizePipe
from pipeline import Pipeline

stream = redisstream.RedisStream()
stream.startListening()

pipeline = Pipeline().addHandler(Base64ToImagePipe) \
                     .addHandler(ImageToCV2Pipe) \
                     .addHandler(CV2CropFacePipe) \
                     .addHandler(CV2ResizePipe) \
                     .addHandler(CV2ToImagePipe) 

def main():
    print('Start listening...')
    while True:
        # Extrage din coada. Daca coada este goala, ramane blocat pana cand apare un element
        try:
            key = stream.queue.get_nowait()
            data = stream.getKeyData(key)
            print(f'Received {key}.')

            image = pipeline.execute(data[0])
            distribution = json.loads(data[1].decode())

            if image is None:
                continue

            print(distribution)
            image.show()

            del image 
            del distribution
            gc.collect()
            stream.deleteKeyData(key)
        except queue.Empty:
            time.sleep(1)

if __name__ == '__main__':
    try:
        main()
    except:
        print(traceback.format_exc())
        print('Exit')
        stream.stopListening()