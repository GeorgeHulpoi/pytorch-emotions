import inspect
from typing import Any, Union
from pipes import PipeTransform

class Pipeline:
    def __init__(self):
        self.__pipes = []
        self.__executing = False

    def pipe(self, pipe: Union[PipeTransform, type]) -> Any:
        if self.__executing is True:
            raise('Pipeline is already executing!')

        if inspect.isclass(pipe) and issubclass(pipe, PipeTransform):
            self.__pipes.append(pipe())
        elif isinstance(pipe, PipeTransform):
            self.__pipes.append(pipe)
        return self

    def execute(self, input: Any) -> Any:
        self.__executing = True
        output = input

        for pipe in self.__pipes:
            output = pipe.transform(output)
            if output is None:
                return None

        self.__executing = False 
        return output