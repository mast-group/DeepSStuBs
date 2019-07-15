import model.AbstractModel

from AbstractModel import *


class ELMoModel(AbstractModel):

    def __init__(self, name_to_vector_file):
        """[summary]
        Loads and instantiates a word2vec model from a json file.
        Arguments:
            name_to_vector_file {[type]} -- [description]
        """
        super().__init__(name_to_vector_file)
        pass
    

    def get_name_to_vector(self):
        """[summary]
        
        Returns:
            [type] -- [description]
        """
        pass
    

    def get_embedding(self, word):
        """[summary]
        Retrieves and returns the FastText embedding of the specified word.
        If the word is out-of-vocabulary then FastText utilizes character 3-gram embedding to calculate one.
        Arguments:
            word {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        pass


    def get_sequence_embeddings(self, sequence):
        """[summary]
        
        Arguments:
            sequence {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        pass
    

    def get_embedding_dims(self):
        """[summary]
        
        Returns:
            [type] -- [description]
        """
        pass
