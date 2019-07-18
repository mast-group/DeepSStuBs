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

        pass


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



if __name__ == "__main__":
    # Now we can compute embeddings.
    raw_code = [
        'STD:function STD:( ID:e STD:, ID:tags STD:) STD:{ ID:tags STD:. ID:should STD:. ' + 
        'ID:have STD:. ID:lengthOf STD:( LIT:1 STD:) STD:; ID:done STD:( STD:) STD:; STD:} '
        'STD:) STD:; STD:} STD:) STD:; STD:} STD:) STD:; STD:} STD:) STD:; STD:} STD:) STD:; ' +
        'STD:} STD:) STD:; STD:} STD:) STD:; STD:} STD:) STD:; STD:} STD:) STD:; STD:} STD:) ' +
        'STD:; STD:} STD:) STD:;',
        'STD:var ID:gfm STD:= ID:require STD:( LIT:github-flavored-markdown STD:) STD:;'
    ]
    tokenized_code = [sentence.split() for sentence in raw_code]
    # tokenized_question = [
    #     ['What', 'are', 'biLMs', 'useful', 'for', '?'],
    # ]
    
    with tf.Session() as sess:
        data_dir = '/disk/scratch/mpatsis/eddie/data/phog/js/'
        model_dir = '/disk/scratch/mpatsis/eddie/models/phog/js/elmo/1024/'
        vocab_file = os.path.join(data_dir, 'vocab')
        weight_file = os.path.join(model_dir, 'weights/weights.hdf5')
        options_file = os.path.join(model_dir, 'query_options.json')
        
        elmo = ELMoModel(model_dir, vocab_file, weight_file, options_file, sess)

        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())

        get_sequence_embeddings(raw_code)

