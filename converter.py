import io
import sys
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
        return torchvision.transforms.ToTensor()(pil)

    def base64ImageToTensor(data):
        return Converter.pilToTensor(Converter.base64ToPil(data))