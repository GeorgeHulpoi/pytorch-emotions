import json
from typing import Any
from pipes import PipeTransform

class JsonParserPipe(PipeTransform):
    def transform(self, input: str) -> Any:
        return json.loads(input)