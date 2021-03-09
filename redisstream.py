import os
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
        self.connectionPool = redis.Redis(host=os.getenv('REDIS_HOST'), port=os.getenv('REDIS_PORT'))
        self.subject= self.connectionPool.pubsub()
        self.subject.subscribe(os.getenv('REDIS_CHANNEL'))

    def threadCallback(self):
        while True:
            if self.threadBreak:
                break
            message = self.subject.get_message()
            if message:
                if  message['type'] == 'message':
                    if message['channel'].decode() == os.getenv('REDIS_CHANNEL'):
                        key = message['data'].decode()
                        # Pune in coada, iar daca este plina (ceea ce nu e cazul aici)
                        # nu va trece mai departe pana cand nu se executa punerea
                        self.queue.put(key)
            time.sleep(0.001)

    def getKeyData(self, key):
        pipe = self.connectionPool.pipeline()
        pipe.hget(key, 'image')
        pipe.hget(key, 'distribution')
        return pipe.execute()

    def startListening(self):
        self.threadBreak = False
        self.thread = threading.Thread(target=self.threadCallback)
        self.thread.start()

    def stopListening(self):
        self.threadBreak = True
