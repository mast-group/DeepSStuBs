from abc import ABC, abstractmethod

class AbstractModel(ABC):

    def __init__(self, path):
        self.path = path
        self.UNK = 'UNK'
        super().__init__()
    

    @abstractmethod
    def get_embedding(self, word):
        """[summary]
        
        Arguments:
            word {[type]} -- [description]
        """
        pass
    
    @abstractmethod
    def get_sequence_embeddings(self, sequence):
        """[summary]
        
        Arguments:
            sequence {[type]} -- [description]
        """
        pass

    @abstractmethod
    def get_embedding_dims(self):
        pass

    def get_UNK(self):
        return self.UNK

