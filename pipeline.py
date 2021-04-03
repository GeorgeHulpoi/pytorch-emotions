from typing import Any
from pipes import Pipe 

class Pipeline:
    def __init__(self):
        self.__handlers = []
        self.__executing = False

    def addHandler(self, handler: Pipe) -> Any:
        if self.__executing is True:
            raise('Pipeline is already executing!')

        self.__handlers.append(handler)
        return self

    def execute(self, input: Any) -> Any:
        self.__executing = True
        output = input

        for handler in self.__handlers:
            output = handler.process(output)
            if output is None:
                return None

        self.__executing = False 
        return output