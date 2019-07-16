import json
import model.AbstractModel

from AbstractModel import *


class Word2VecModel(AbstractModel):

    def __init__(self, name_to_vector_file):
        """[summary]
        Loads and instantiates a word2vec model from a json file.
        Arguments:
            name_to_vector_file {[type]} -- [description]
        """
        super().__init__(name_to_vector_file)
        assert(name_to_vector_file.endswith('.json'))
        with open(name_to_vector_file) as f:
            self.name_to_vector = json.load(f)
    

    def get_name_to_vector(self):
        """[summary]
        
        Returns:
            [type] -- [description]
        """
        return self.name_to_vector
    

    def get_embedding(self, word):
        """[summary]
        Retrieves and returns the word2vec embedding of the specified word.
        If the word is out-of-vocabulary the embedding for UNK is returned instead.
        Arguments:
            word {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        if word in self.name_to_vector:
            return self.name_to_vector[word]
        else:
            return self.name_to_vector[self.UNK]
    

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
        return word == self.get_UNK() or not word in self.name_to_vector

