import torch
import PIL.Image
import torchvision.transforms
from pipes import PipeTransform

class ImageToTensorPipe(PipeTransform):
    def transform(self, input: PIL.Image) -> torch.tensor:
        return torchvision.transforms.ToTensor()(input).unsqueeze(0)