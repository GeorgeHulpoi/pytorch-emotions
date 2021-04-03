import abc

class Pipe(abc.ABC):

    @abc.abstractmethod
    def process(self, input):
        return