import time
import pickle
import queue
import settings
import redisstream

stream = redisstream.RedisStream()
stream.startListening()
emotions = []

def main():
    while True:
        # Extrage din coada. Daca coada este goala, ramane blocat pana cand apare un element
        try:
            key = stream.queue.get_nowait()
            data = stream.getKeyData(key)
            print(f'Received {key}.')
            emotions.append(data)
        except queue.Empty:
            time.sleep(1)

def save():
    pickle.dump(emotions, open('data.pickle', 'wb'))

if __name__ == '__main__':
    try:
        main()
    except (SystemExit, KeyboardInterrupt):
        print('Exit')
        stream.stopListening()
        save()