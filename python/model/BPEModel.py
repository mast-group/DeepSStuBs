import json
import model.AbstractModel
import tensorflow as tf

from AbstractModel import *
from codenlm import reader
from codenlm import code_nlm


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
        
        init_scale = 0.05
        learning_rate = 0.1
        max_grad_norm = 5.0
        num_layers = 1
        num_steps = 20
        hidden_size = 512
        max_epoch = 30
        keep_prob = 1.0
        lr_decay = 0.5
        batch_size = 32
        test_batch_size = 10
        
        config = Config( init_scale, learning_rate, max_grad_norm, num_layers, num_steps, hidden_size, 
                        max_epoch, keep_prob, lr_decay, batch_size, test_batch_size, len(train_vocab) )
        
        # config = code_nlm.Config(code_nlm.FLAGS.init_scale, code_nlm.FLAGS.learning_rate, code_nlm.FLAGS.max_grad_norm,
        #             code_nlm.FLAGS.num_layers, code_nlm.FLAGS.num_steps, code_nlm.FLAGS.hidden_size, 
        #             code_nlm.FLAGS.max_epoch, code_nlm.FLAGS.keep_prob, code_nlm.FLAGS.lr_decay, 
        #             code_nlm.FLAGS.batch_size, code_nlm.FLAGS.test_batch_size, 
        #             code_nlm.FLAGS.vocab_size, code_nlm.FLAGS.output_probs_file)
        # config.vocab_size = len(train_vocab)

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


class Config(object):
  """Configuration"""

  def __init__(self, inits, lr, mgrad, nlayers, nsteps, hsize, mepoch, kp, 
                decay, bsize, tbsize, vsize):
    self.init_scale = inits
    self.learning_rate = lr
    self.max_grad_norm = mgrad
    self.num_layers = nlayers
    self.num_steps = nsteps
    self.hidden_size = hsize
    self.max_epoch = mepoch
    self.keep_prob = kp
    self.lr_decay = decay
    self.batch_size = bsize
    self.test_batch_size = tbsize
    self.vocab_size = vsize
