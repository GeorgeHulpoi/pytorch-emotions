import torch
from typing import Any
from pipes import Pipe

class DistributionToLabelPipe(Pipe):
    def process(self, input: Any) -> int:
        output = [float(input['angry']), float(input['happy']), float(input['surprise'])]
        output = torch.FloatTensor(output)
        return torch.argmax(output).unsqueeze(0)