import model.AbstractModel
import tensorflow as tf
import os

from AbstractModel import *
from bilm import Batcher, BidirectionalLanguageModel, weight_layers


class ELMoModel(AbstractModel):

    def __init__(self, model_file, vocab_file, weight_file, options_file, sess, threshold=50):
        super().__init__(model_file)
        # Create a Batcher to map text to character ids.
        self._batcher = Batcher(vocab_file, 50)
        # Input placeholders to the biLM.
        code_character_ids = tf.placeholder('int32', shape=(None, None, 50))
        self._weight_file = weight_file
        self._options_file = options_file
        
        # Build the biLM graph.
        bilm = self._create_bilm()
        # Get ops to compute the LM embeddings.
        code_embeddings_op = bilm(code_character_ids)
        self.elmo_code_rep_op = weight_layers('input', code_embeddings_op, l2_coef=0.0)


    def _create_bilm(self):
         # Build the biLM graph.
        bilm = BidirectionalLanguageModel(
            self._options_file,
            self._weight_file,
            use_character_inputs=True
        )
        return bilm


    # def __init__(self, sess, batcher, elmo_token_op, code_token_ids, threshold=30, dims=200):
    #     self.sess = sess
    #     self.batcher = batcher
    #     self.elmo_token_op = elmo_token_op
    #     self.code_token_ids = code_token_ids
    #     self.threshold = threshold
    #     self.dims = dims
    

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
        # Create batches of data.
        code_ids = self._batcher.batch_sentences(tokenized_code)
        # question_ids = batcher.batch_sentences(tokenized_question)

        # Warm up the LSTM state, otherwise will get inconsistent embeddings.
        for step in range(500):
            elmo_code_representation = sess.run(
                [elmo_code_rep_op['weighted_op']],
                feed_dict={code_character_ids: code_ids}
            )

        # Compute ELMo representations (here for the input only, for simplicity).
        elmo_code_representation = sess.run(
            [elmo_code_rep_op['weighted_op']],
            feed_dict={code_character_ids: code_ids}
        )
        print(elmo_code_representation)
    

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


