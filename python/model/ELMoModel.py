import glob
import model.AbstractModel
import tensorflow as tf
import os

from AbstractModel import *
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
from bilm.data import BidirectionalLMDataset
from bilm.training import load_vocab


class ELMoModel(AbstractModel):

    def __init__(self, model_file, vocab_file, weight_file, options_file, sess, threshold=50):
        super().__init__(model_file)
        # Create a Batcher to map text to character ids.
        self._vocab_file = vocab_file
        self._batcher = Batcher(vocab_file, 50)
        # Input placeholders to the biLM.
        self._code_character_ids = tf.placeholder('int32', shape=(None, None, 50))
        self._weight_file = weight_file
        self._options_file = options_file
        self._sess = sess
        
        # Build the biLM graph.
        bilm = self._create_bilm()
        # Get ops to compute the LM embeddings.
        code_embeddings_op = bilm(self._code_character_ids)
        self._elmo_code_rep_op = weight_layers('input', code_embeddings_op, l2_coef=0.0)


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
        code_ids = self._batcher.batch_sentences(sequence)

        # # Warm up the LSTM state, otherwise will get inconsistent embeddings.
        # for step in range(500):
        #     elmo_code_representation = self._sess.run(
        #         [self._elmo_code_rep_op['weighted_op']],
        #         feed_dict={self._code_character_ids: code_ids}
        #     )

        # Compute ELMo representations (here for the input only, for simplicity).
        elmo_code_representation = self._sess.run(
            [self._elmo_code_rep_op['weighted_op']],
            feed_dict={self._code_character_ids: code_ids}
        )
        print(elmo_code_representation)
        return elmo_code_representation
    

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

    
    def _batch_generator(self, data_prefix, batch_size, sentence_len):
        batch = []

        files = self._all_shards = glob.glob(data_prefix)
        for file in files:
            print(file)
            with open(file, 'r') as f:
                for program in f:
                    program = program.split()
                    program_sequences = [program[i * sentence_len:(i + 1) * sentence_len] \
                        for i in range((len(program) + sentence_len - 1) // sentence_len )]
                    for program_sequence in program_sequences:
                        print(len(program_sequence), program_sequence)
                        batch.append(' '.join(program_sequence))
                        if len(batch) == batch_size:
                            yield batch
                            batch = []
                            print()
                            break
                    break
            break
        # if len(batch) > 0:
        #     yield batch


    def warm_up(self, data_prefix):
        """[summary]
        
        Arguments:
            data_prefix {[type]} -- [description]
        """
        batch_size = 16
        sequence_size = 50
        vocab = load_vocab(self._vocab_file, 50)

        for batch in self._batch_generator(data_prefix, batch_size, sequence_size):
            print(batch)
            embeddings = self.get_sequence_embeddings(batch)
            print(len(batch))
            print(batch[0].shape)
        
        # data = BidirectionalLMDataset(data_prefix, vocab, test=False, shuffle_on_load=False)
        # for batch in data.iter_batches(batch_size, 20):
        #     print(len(batch))
        #     print(batch)
        #     break
        pass

