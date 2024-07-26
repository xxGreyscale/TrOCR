# Make an interface for the loader class
from abc import abstractmethod, ABC


class LoaderInterface(ABC):
    def __init__(self, path=None):
        self.path = path

    @abstractmethod
    def generate_dataframe(self):
        pass

    @abstractmethod
    def calculate_max_character_length(self):
        pass

    @abstractmethod
    def get_dataframe(self):
        pass

    @abstractmethod
    def get_maximum_char_length(self):
        pass

    @abstractmethod
    def generate_histogram(self):
        pass
