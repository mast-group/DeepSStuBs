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
        # code_embeddings_op is a dict containing various tensor ops.
        code_embeddings_op = bilm(self._code_character_ids)
        print('code_embeddings_op:', code_embeddings_op)
        
        lm_embeddings = code_embeddings_op['lm_embeddings']
        self._n_lm_layers = int(lm_embeddings.get_shape()[1])
        self._emb_dims = int(lm_embeddings.get_shape()[3])
        # self._emb_dims = code_embeddings_op['lm_embeddings'].get_shape().as_list()[-1]
        

        # Create a token level representation op
        self._token_rep_op = weight_layers('token', code_embeddings_op, l2_coef=0.0, use_tokens_only=True, use_top_only=False)
        
        # Create an ELMo top-layer only representation op
        self._elmo_top_rep_op = weight_layers('top', code_embeddings_op, l2_coef=0.0, use_top_only=True)
        
        # Create an ELMo representation op
        self._elmo_code_rep_op = weight_layers('elmo', code_embeddings_op, l2_coef=0.0, use_top_only=False, is_trainable=True)
        print('_elmo_code_rep_op:', self._elmo_code_rep_op)
        


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
    

    def get_embedding(self, word, token=True):
        """[summary]
        Retrieves and returns the token embedding of the specified word.
        Arguments:
            word {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        code_ids = self._batcher.batch_sentences([[word]])

        if token: 
            elmo_token_representation = self._sess.run(
                [self._token_rep_op['weighted_op']],
                feed_dict={self._code_character_ids: code_ids}
            )
            return elmo_token_representation
        else:
            elmo_code_representation = self._sess.run(
                [self._elmo_code_rep_op['weighted_op']],
                feed_dict={self._code_character_ids: code_ids}
            )
            return elmo_code_representation


    def get_top_embedding(self, word):
        """[summary]
        Retrieves and returns the token embedding of the specified word.
        Arguments:
            word {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        code_ids = self._batcher.batch_sentences([[word]])
        elmo_top_representation = self._sess.run(
            [self._elmo_top_rep_op['weighted_op']],
            feed_dict={self._code_character_ids: code_ids}
        )
        return elmo_top_representation


    def get_sequence_embeddings(self, sequence):
        """[summary]
        
        Arguments:
            sequence {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        # Create batches of data.
        code_ids = self._batcher.batch_sentences(sequence)

        # Compute ELMo representations.
        elmo_code_representation = self._sess.run(
            [self._elmo_code_rep_op['weighted_op']],
            feed_dict={self._code_character_ids: code_ids}
        )
        return elmo_code_representation[0]
    

    def get_sequence_default_embeddings(self, sequence):
        """[summary]
        
        Arguments:
            sequence {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        # Create batches of data.
        code_ids = self._batcher.batch_sentences(sequence)

        # Compute ELMo representations.
        elmo_code_representation = self._sess.run(
            [self._elmo_code_rep_op['weighted_op']],
            feed_dict={self._code_character_ids: code_ids}
        )
        return elmo_code_representation[0]
    

    def get_sequence_token_embeddings(self, sequence):
        """[summary]
        
        Arguments:
            sequence {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        # Create batches of data.
        code_ids = self._batcher.batch_sentences(sequence)
        # code_ids contains character ids for each token.
        
        # Compute token layer representations.
        elmo_token_representation = self._sess.run(
            [self._token_rep_op['weighted_op']],
            feed_dict={self._code_character_ids: code_ids}
        )
        return elmo_token_representation[0]
    

    def get_sequence_top_embeddings(self, sequence):
        """[summary]
        
        Arguments:
            sequence {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        # Create batches of data.
        code_ids = self._batcher.batch_sentences(sequence)
        
        # Compute top layer representations.
        elmo_top_representation = self._sess.run(
            [self._elmo_top_rep_op['weighted_op']],
            feed_dict={self._code_character_ids: code_ids}
        )
        return elmo_top_representation[0]
    

    def get_embedding_dims(self):
        """[summary]
        
        Returns:
            [type] -- [description]
        """
        return self._emb_dims
    

    def get_token_embedding_dims(self):
        """[summary]
        
        Returns:
            [type] -- [description]
        """
        return self._emb_dims / 2
    

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
                        # print(len(program_sequence), program_sequence)
                        batch.append(program_sequence)
                        # batch.append(' '.join(program_sequence))
                        if len(batch) == batch_size:
                            yield batch
                            batch = []
                            # print()
        if len(batch) > 0:
            yield batch


    def warm_up(self, data_prefix):
        """[summary]
        
        Arguments:
            data_prefix {[type]} -- [description]
        """
        batch_size = 32
        sequence_size = 50
        vocab = load_vocab(self._vocab_file, 50)

        for batch in self._batch_generator(data_prefix, batch_size, sequence_size):
            # print(batch)
            embeddings = self.get_sequence_embeddings(batch)
            # print(len(embeddings))
            # print(embeddings[0].shape)
            # break
        
        # data = BidirectionalLMDataset(data_prefix, vocab, test=False, shuffle_on_load=False)
        # for batch in data.iter_batches(batch_size, 20):
        #     print(len(batch))
        #     print(batch)
        #     break
        pass

