import PIL.Image
import torchvision.transforms

from pipes import Pipe

class ImageToTensorPipe(Pipe):
    def process(self, input: PIL.Image) -> PIL.Image:
        return torchvision.transforms.ToTensor()(input).unsqueeze(0)