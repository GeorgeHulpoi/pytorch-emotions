import io
import base64
import PIL.Image
from pipes import PipeTransform

class Base64ToImagePipe(PipeTransform):
    def transform(self, input: str) -> PIL.Image:
        bytes = base64.decodebytes(input)
        return PIL.Image.open(io.BytesIO(bytes))