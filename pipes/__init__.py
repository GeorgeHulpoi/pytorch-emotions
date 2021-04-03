from pipes.pipe import Pipe

from pipes.base64_to_image import Base64ToImagePipe as _Base64ToImagePipe
from pipes.image_to_tensor import ImageToTensorPipe as _ImageToTensorPipe
from pipes.image_to_cv2 import ImageToCV2Pipe as _ImageToCV2Pipe
from pipes.cv2_crop_face import CV2CropFacePipe as _CV2CropFacePipe
from pipes.cv2_to_tensor import CV2ToTensorPipe as _CV2ToTensorPipe
from pipes.cv2_to_image import CV2ToImagePipe as _CV2ToImagePipe
from pipes.json_parser import JsonParserPipe as _JsonParserPipe
from pipes.distribution_to_label import DistributionToLabelPipe as _DistributionToLabelPipe
from pipes.cv2_resize import CV2ResizePipe as _CV2ResizePipe

Base64ToImagePipe = _Base64ToImagePipe()
ImageToTensorPipe = _ImageToTensorPipe()
ImageToCV2Pipe = _ImageToCV2Pipe()
CV2CropFacePipe = _CV2CropFacePipe()
CV2ToTensorPipe = _CV2ToTensorPipe()
CV2ToImagePipe = _CV2ToImagePipe()
JsonParserPipe = _JsonParserPipe()
DistributionToLabelPipe = _DistributionToLabelPipe()
CV2ResizePipe = _CV2ResizePipe()