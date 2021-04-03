import io
import base64
import PIL.Image

from pipes import Pipe

class Base64ToImagePipe(Pipe):
    def process(self, input: str) -> PIL.Image:
        bytes = base64.decodebytes(input)
        return PIL.Image.open(io.BytesIO(bytes))