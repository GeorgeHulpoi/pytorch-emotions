import io
import sys
import json
import torch
import base64
import PIL.Image
import torchvision.transforms

if __name__ == "__main__":
    sys.exit('Don\'t run as main script!')

class Converter:
    def base64ToPil(data):
        bytes = base64.decodebytes(data)
        return PIL.Image.open(io.BytesIO(bytes))

    def pilToTensor(pil):
        return torchvision.transforms.ToTensor()(pil).unsqueeze(0)

    def base64ImageToTensor(data):
        return Converter.pilToTensor(Converter.base64ToPil(data))

    def distributionToLabel(distribution):
        vector = [float(distribution['angry']), float(distribution['happy']), float(distribution['surprise'])]
        distribution = torch.FloatTensor(vector)
        return torch.argmax(distribution).unsqueeze(0)

    def dataToTensor(data):
        distribution = json.loads(data[1].decode())
        label = Converter.distributionToLabel(distribution)
        image = Converter.base64ImageToTensor(data[0])

        return (image, label)

