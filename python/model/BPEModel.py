import json
import model.AbstractModel
import tensorflow as tf

from AbstractModel import *
from codenlm import reader
from codenlm import code_nlm
from absl


class BPEModel(AbstractModel):

    def __init__(self, model_file, vocab_file, sess):
        """[summary]
        Loads and instantiates a word2vec model from a json file.
        Arguments:
            name_to_vector_file {[type]} -- [description]
        """
        super().__init__(model_file)

        self._vocab_file = vocab_file
        train_vocab, train_vocab_rev = reader._read_vocab(vocab_file)
        self._sess = sess

        try:
            config = code_nlm.Config(code_nlm.FLAGS.init_scale, code_nlm.FLAGS.learning_rate, code_nlm.FLAGS.max_grad_norm,
                        code_nlm.FLAGS.num_layers, code_nlm.FLAGS.num_steps, code_nlm.FLAGS.hidden_size, 
                        code_nlm.FLAGS.max_epoch, code_nlm.FLAGS.keep_prob, code_nlm.FLAGS.lr_decay, 
                        code_nlm.FLAGS.batch_size, code_nlm.FLAGS.test_batch_size, 
                        code_nlm.FLAGS.vocab_size, code_nlm.FLAGS.output_probs_file)
        except Exception:
            print('exception')
        config.vocab_size = len(train_vocab)

        with tf.Graph().as_default():
            self.model = code_nlm.create_model(self._sess, config)
            self.model.train_vocab = train_vocab
            self.model.train_vocab_rev = train_vocab_rev
            feed_dict = {
                self.inputd: 100,
                self.keep_probability: 1.0
            }
            embedded_inputds = self._sess.run([model.embedded_inputds], feed_dict)
            print('Queried for embedded inputs')
            print(embedded_inputds)


    def get_name_to_vector(self):
        """[summary]
        
        Returns:
            [type] -- [description]
        """
        raise AttributeError
    

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

