import gc
import time
import queue
import traceback
import redisstream
from pipes import Pipeline, Base64ToImagePipe, ImageToCV2Pipe, CV2CropFacePipe, CV2ToImagePipe, CV2ResizePipe, JsonParserPipe

stream = redisstream.RedisStream()
stream.startListening()

imagePipeline = Pipeline().pipe(Base64ToImagePipe) \
                          .pipe(ImageToCV2Pipe) \
                          .pipe(CV2CropFacePipe) \
                          .pipe(CV2ResizePipe) \
                          .pipe(CV2ToImagePipe)

labelPipeline = Pipeline().pipe(JsonParserPipe)

def main():
    while True:
        try:
            key = stream.queue.get_nowait()
            data = stream.getKeyData(key)
            print(f'Received {key}.')

            image = imagePipeline.execute(data[0])
            distribution = labelPipeline.execute(data[1].decode())

            if image is not None:
                print(distribution)
                image.show()

            del image 
            del distribution
            gc.collect()
            stream.deleteKeyData(key)
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