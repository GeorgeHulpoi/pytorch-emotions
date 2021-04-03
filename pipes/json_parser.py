import json
from typing import Any
from pipes import Pipe

class JsonParserPipe(Pipe):
    def process(self, input: str) -> Any:
        return json.loads(input)