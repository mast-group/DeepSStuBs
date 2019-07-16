import model.AbstractModel

from AbstractModel import *
from gensim.models import FastText


class FastTextModel(AbstractModel):

    def __init__(self, name_to_vector_file):
        """[summary]
        Loads and instantiates a word2vec model from a json file.
        Arguments:
            name_to_vector_file {[type]} -- [description]
        """
        super().__init__(name_to_vector_file)
        assert(name_to_vector_file.endswith('.json'))
        self.m = FastText.load(name_to_vector_file)
        self.name_to_vector = m.wv
    

    def get_name_to_vector(self):
        """[summary]
        
        Returns:
            [type] -- [description]
        """
        return self.name_to_vector
    

    def get_embedding(self, word):
        """[summary]
        Retrieves and returns the FastText embedding of the specified word.
        If the word is out-of-vocabulary then FastText utilizes character 3-gram embedding to calculate one.
        Arguments:
            word {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        return list(self.name_to_vector[word])
    

    def get_sequence_embeddings(self, sequence):
        """[summary]
        
        Arguments:
            sequence {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        assert(type(sequence) == list)
        embeddings = [self.get_embedding(word) for word in sequence]
        return embeddings
    

    def get_embedding_dims(self):
        """[summary]
        
        Returns:
            [type] -- [description]
        """
        return len(self.name_to_vector[self.UNK])
    

    def isOOV(self, word):
        """[summary]
        
        Arguments:
            word {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        return False

