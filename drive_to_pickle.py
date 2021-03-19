import base64
import pickle
from PIL import Image
from io import BytesIO

labels = [
    0,
    0,
    0,
    1,
    -1,
    0,
    0,
    1,
    2,
    1,
    -1,
    2,
    2,
    1,
    1,
    2,
    1,
    0,
    2,
    0,
    1,
    0,
    -1,
    1,
    0,
    2,
    -1,
    1,
    0,
    2,
    2,
    0,
    2,
    -1,
    2,
    0,
    0,
    1,
    1,
    1
]

images = []

for i in range(1, 41):
    img = Image.open(f'test/{i}.png')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    images.append(img_str)

data = list(zip(images, labels))
pickle.dump(data, open('test.pickle', 'wb'))
