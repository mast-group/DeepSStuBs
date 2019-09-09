import model.AbstractModel
import tensorflow as tf
import os

from AbstractModel import *
from bilm import Batcher, BidirectionalLanguageModel, weight_layers


class ELMoBPEModel(AbstractModel):

    def __init__(self, name_to_vector_file):
        """[summary]
        Loads and instantiates a word2vec model from a json file.
        Arguments:
            name_to_vector_file {[type]} -- [description]
        """
        super().__init__(name_to_vector_file)
        pass
    

    def __init__(self, model_file, options_file, sess, threshold=50):
        super().__init__(model_file)
        self.options_file = options_file
        pass

    def _create_bilm(self, options_file, weight_file):
         # Build the biLM graph.
        bilm = BidirectionalLanguageModel(
            options_file,
            weight_file,
            use_character_inputs=True
        )
        pass


    def __init__(self, sess, batcher, elmo_token_op, code_token_ids, threshold=30, dims=200):
        self.sess = sess
        self.batcher = batcher
        self.elmo_token_op = elmo_token_op
        self.code_token_ids = code_token_ids
        self.threshold = threshold
        self.dims = dims
    

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
    

    def isOOV(self, word):
        """[summary]
        
        Arguments:
            word {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        return False

