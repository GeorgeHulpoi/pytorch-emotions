import time
import pickle
import queue
import settings
import redisstream

stream = redisstream.RedisStream()
stream.startListening()
keys = []

def main():
    while True:
        try:
            key = stream.queue.get_nowait()
            print(f'Received {key}.')
            keys.append(key)
        except queue.Empty:
            time.sleep(1)

def save():
    pickle.dump(keys, open('keys.pickle', 'wb'))

if __name__ == '__main__':
    try:
        main()
    except (SystemExit, KeyboardInterrupt):
        print('Exit')
        stream.stopListening()
        save()