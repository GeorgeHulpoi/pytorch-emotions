import abc

class PipeTransform(abc.ABC):

    @abc.abstractmethod
    def transform(self, input):
        return