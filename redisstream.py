import sys
import time
import queue
import threading
import redis

if __name__ == "__main__":
    sys.exit('Don\'t run as main script!')

class RedisStream:

    def __init__(self):
        self.queue = queue.Queue()
        self.__connectionPool = redis.Redis(host='localhost', port=6379)
        self.__subject = self.__connectionPool.pubsub()
        self.__subject.subscribe('dataset-generator')

    def deleteKeyData(self, key):
        pipe = self.__connectionPool.pipeline()
        pipe.hdel(key, 'image')
        pipe.hdel(key, 'distribution')
        return pipe.execute()

    def getKeyData(self, key):
        pipe = self.__connectionPool.pipeline()
        pipe.hget(key, 'image')
        pipe.hget(key, 'distribution')
        return pipe.execute()

    def startListening(self):
        self.__threadBreak = False
        self.__thread = threading.Thread(target=self.__threadCallback)
        self.__thread.start()

    def stopListening(self):
        self.__threadBreak = True

    def __threadCallback(self):
        while True:
            if self.__threadBreak:
                break
            message = self.__subject.get_message()
            if message:
                if  message['type'] == 'message':
                    if message['channel'].decode() == 'dataset-generator':
                        key = message['data'].decode()
                        self.queue.put(key)
            time.sleep(0.01)
