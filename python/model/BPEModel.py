import json
import model.AbstractModel
import tensorflow as tf
import inspect
import numpy as np

# BPE imports
import codecs
from subword_nmt.apply_bpe import BPE, read_vocabulary

from AbstractModel import *
from codenlm import reader
# from codenlm import code_nlm


def data_type():
  """
  Returns the TF floating point type used for operations.
  :return: The data type used (tf.float32)
  """
  return tf.float32

def get_gpu_config():
  gconfig = tf.ConfigProto()
  gconfig.gpu_options.per_process_gpu_memory_fraction = 0.975 # Don't take 100% of the memory
  gconfig.allow_soft_placement = True # Does not aggressively take all the GPU memory
  gconfig.gpu_options.allow_growth = True # Take more memory when necessary
  return gconfig

class NLM(object):

  def __init__(self, config):
    """
    Initializes the neural language model based on the specified configation.
    :param config: The configuration to be used for initialization.
    """
    self.num_layers = config.num_layers
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.hidden_size = hidden_size = config.hidden_size
    self.vocab_size = vocab_size = config.vocab_size
    self.global_step = tf.Variable(0, trainable=False)
    self.gru = config.gru

    with tf.name_scope("Parameters"):
      # Sets dropout and learning rate.
      self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
      self.keep_probability = tf.placeholder(tf.float32, name="keep_probability")

    with tf.name_scope("Input"):
      self.inputd = tf.placeholder(tf.int64, shape=(batch_size, None), name="inputd")
      self.targets = tf.placeholder(tf.int64, shape=(batch_size, None), name="targets")
      self.target_weights = tf.placeholder(tf.float32, shape=(batch_size, None), name="tgtweights")

    with tf.device("/cpu:0"):
      with tf.name_scope("Embedding"):
        # Initialize embeddings on the CPU and add dropout layer after embeddings.
        self.embedding = tf.Variable(tf.random_uniform((vocab_size, hidden_size), -config.init_scale, config.init_scale), dtype=data_type(), name="embedding")
        self.embedded_inputds = tf.nn.embedding_lookup(self.embedding, self.inputd, name="embedded_inputds")
        self.embedded_inputds = tf.nn.dropout(self.embedded_inputds, self.keep_probability)

    with tf.name_scope("RNN"):
      # Definitions for the different cells that can be used. Either lstm or GRU which will be wrapped with dropout.
      def lstm_cell():
        if 'reuse' in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args:
          return tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        else:
          return tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
      def gru_cell():
        if 'reuse' in inspect.getargspec(tf.contrib.rnn.GRUCell.__init__).args:
          return tf.contrib.rnn.GRUCell(hidden_size, reuse=tf.get_variable_scope().reuse)
        else:
          return tf.contrib.rnn.GRUCell(hidden_size)
      def drop_cell():
        if config.gru:
          return tf.contrib.rnn.DropoutWrapper(gru_cell(), output_keep_prob=self.keep_probability)
        else:
          return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=self.keep_probability)

      # Allows multiple layers to be used. Not advised though.
      rnn_layers = tf.contrib.rnn.MultiRNNCell([drop_cell() for _ in range(self.num_layers)], state_is_tuple=True)
      # Initialize the state to zero.
      self.reset_state = rnn_layers.zero_state(batch_size, data_type())
      self.outputs, self.next_state = tf.nn.dynamic_rnn(rnn_layers, self.embedded_inputds, time_major=False,
                                                        initial_state=self.reset_state)

    with tf.name_scope("Cost"):
      self.output = tf.reshape(tf.concat(axis=0, values=self.outputs), [-1, hidden_size])
      self.softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size], dtype=data_type())
      self.softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
      self.logits = tf.matmul(self.output, self.softmax_w) + self.softmax_b
      self.loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [self.logits], [tf.reshape(self.targets, [-1])], [tf.reshape(self.target_weights, [-1])])
      self.cost = tf.div(tf.reduce_sum(self.loss), batch_size, name="cost")
      self.final_state = self.next_state

      self.norm_logits = tf.nn.softmax(self.logits)

    with tf.name_scope("Train"):
      self.iteration = tf.Variable(0, dtype=data_type(), name="iteration", trainable=False)
      tvars = tf.trainable_variables()
      self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                                 config.max_grad_norm, name="clip_gradients")
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.train_step = optimizer.apply_gradients(zip(self.gradients, tvars), name="train_step",
                                                  global_step=self.global_step)
      self.validation_perplexity = tf.Variable(dtype=data_type(), initial_value=float("inf"),
                                               trainable=False, name="validation_perplexity")
      tf.summary.scalar(self.validation_perplexity.op.name, self.validation_perplexity)
      self.training_epoch_perplexity = tf.Variable(dtype=data_type(), initial_value=float("inf"),
                                                   trainable=False, name="training_epoch_perplexity")
      tf.summary.scalar(self.training_epoch_perplexity.op.name, self.training_epoch_perplexity)

      self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
      self.initialize = tf.initialize_all_variables()
      self.summary = tf.summary.merge_all()


  def get_parameter_count(self, debug=False):
    """
    Counts the number of parameters required by the model.
    :param debug: Whether debugging information should be printed.
    :return: Returns the number of parameters required for the model.
    """
    params = tf.trainable_variables()
    total_parameters = 0
    for variable in params:
      shape = variable.get_shape()
      variable_parameters = 1
      for dim in shape:
        variable_parameters *= dim.value
      if debug:
          print(variable)
          print(shape + "\t" + str(len(shape)) + "\t" + str(variable_parameters))
      total_parameters += variable_parameters
    return total_parameters

  @property
  def reset_state(self):
      return self._reset_state

  @reset_state.setter
  def reset_state(self, x):
      self._reset_state = x

  @property
  def cost(self):
      return self._cost

  @cost.setter
  def cost(self, y):
      self._cost = y

  @property
  def final_state(self):
      return self._final_state

  @final_state.setter
  def final_state(self, z):
      self._final_state = z

  @property
  def learning_rate(self):
      return self._lr

  @learning_rate.setter
  def learning_rate(self, l):
      self._lr = l

  @property
  def input(self):
      return self.data



class BPEModel(AbstractModel):

    def __init__(self, model_file, vocab_file, codes_file, merge, sess):
        """[summary]
        Loads and instantiates a BPE model from a tensoflow checkpoint file.
        Arguments:
            name_to_vector_file {[type]} -- [description]
        """
        super().__init__(model_file)
        self._model_dir = model_file
        
        with open(codes_file, 'r') as bpe_codes_fin:
            self._bpe = BPE(bpe_codes_fin, merges=-1, separator='@@')

        self.merge = merge
        self._vocab_file = vocab_file
        train_vocab, train_vocab_rev = reader._read_vocab(vocab_file)
        print(len(train_vocab))
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

        with self._sess.graph.as_default():
        # with tf.Graph().as_default():
            self.model = self.__create_model__(self._sess, config)
            # self.model = self.__create_model__(session, config)
            self.model.train_vocab = train_vocab
            self.model.train_vocab_rev = train_vocab_rev
            feed_dict = {
                self.model.inputd: np.expand_dims(np.array([100] * 32), axis=1),
                self.model.keep_probability: 1.0
            }
            embedded_inputds = self._sess.run([self.model.embedded_inputds], feed_dict)
            # embedded_inputds = session.run([model.embedded_inputds], feed_dict)
            print('Queried for embedded inputs')
            print(embedded_inputds)
        print('Query')
        print(self.get_embedding('publicios'))
        sequences = [['publicios', 'static', 'void'] for i in range(self.model.batch_size)]
        sequences[-1][-1] = 'voidmain'
        sequence_representations = self.get_sequence_token_embeddings(sequences)
        print(sequence_representations, sequence_representations.shape)
        sequence_representations = self.get_sequence_embeddings(sequences)
        print('here')
        print(sequence_representations)
        print(sequence_representations.shape)
    

    def __create_model__(self, session, config):
        """
        Creates the NLM and restores its parameters if there is a saved checkpoint.
        :param session: The TF session in which operations will be run.
        :param config: The configuration to be used.
        :return:
        """
        model = NLM(config)
        ckpt = tf.train.get_checkpoint_state(self._model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
            print('Done restoring the model.')
        else:
            print("Created model with fresh parameters:")
            session.run(tf.global_variables_initializer())
        print("*Number of parameters* = " + str(model.get_parameter_count()))
        return model


    def get_name_to_vector(self):
        """[summary]
        
        Returns:
            [type] -- [description]
        """
        pass
    

    def get_embedding(self, word, token=True):
        """[summary]
        Retrieves and returns the token embedding of the specified word.
        Using this method is not suggested and its sequence variant should be preffered instead.
        Arguments:
            word {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        subtokens = self._bpe.segment(word).split()
        code_ids = [self.model.train_vocab[subtoken] for subtoken in subtokens]
        

        if token:
            with self._sess.graph.as_default():
                feed_dict = {
                    self.model.inputd: np.array([code_ids for i in range(self.model.batch_size)]),
                    # self.model.inputd: np.expand_dims(np.array(code_ids), axis=1),
                    # self.model.inputd: np.expand_dims(np.array([100] * 32), axis=1),
                    self.model.keep_probability: 1.0
                }
                bpe_token_representation = self._sess.run([self.model.embedded_inputds], feed_dict)
                print(len(subtokens))
                print(bpe_token_representation[0][0].shape)
                # print(np.squeeze(bpe_token_representation[0][:len(subtokens)], axis=1))
                return bpe_token_representation[0][0]
        else:
            pass
            # elmo_code_representation = self._sess.run(
            #     [self._elmo_code_rep_op['weighted_op']],
            #     feed_dict={self._code_character_ids: code_ids}
            # )
            # return elmo_code_representation
    

    def get_sequence_token_embeddings(self, sequence):
        """[summary]
        
        Arguments:
            sequence {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        assert(type(sequence) == list)
        assert(len(sequence) == self.model.batch_size)

        code_ids = []
        subword_sizes = []
        for sentence in sequence:
            sentence_ids = []
            subword_sizes.append([])
            for word in sentence:
                subtokens = self._bpe.segment(word).split()
                subword_sizes[-1].append(len(subtokens))
                sentence_ids.extend([self.model.train_vocab[subtoken] for subtoken in subtokens])
            code_ids.append(sentence_ids)
        
        # Find sentence lengths
        sizes = [len(sentence_ids) for sentence_ids in code_ids]
        longest = max(sizes)
        filler_id = self.model.train_vocab['</s>']
        for i in range(len(code_ids)):
            code_ids[i].extend( [filler_id] * (longest - len(code_ids[i])) )

        # print(code_ids)
        # print(np.array(code_ids))

        with self._sess.graph.as_default():
            feed_dict = {
                self.model.inputd: np.array(code_ids),
                self.model.keep_probability: 1.0
            }
            bpe_token_representation = self._sess.run([self.model.embedded_inputds], feed_dict)
            # print(bpe_token_representation[0].shape)

            if True:
                summed_representations = []
                for representation, sub_sizes in zip(bpe_token_representation[0], subword_sizes):
                    summed_representations.append([])
                    index = 0
                    for sub_size in sub_sizes:
                        summed_representations[-1].extend(np.sum(representation[index: index + sub_size], axis=0))
                        index += sub_size
                    # sum()
                return np.array(summed_representations)
            return bpe_token_representation[0]
    

    def get_sequence_embeddings(self, sequence):
        """[summary]
        
        Arguments:
            sequence {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        assert(type(sequence) == list)
        assert(len(sequence) == self.model.batch_size)

        code_ids = []
        subword_sizes = []
        for sentence in sequence:
            sentence_ids = []
            subword_sizes.append([])
            for word in sentence:
                subtokens = self._bpe.segment(word).split()
                subword_sizes[-1].append(len(subtokens))
                sentence_ids.extend([self.model.train_vocab[subtoken] for subtoken in subtokens])
            code_ids.append(sentence_ids)
        
        # Find sentence lengths
        sizes = [len(sentence_ids) for sentence_ids in code_ids]
        longest = max(sizes)
        filler_id = self.model.train_vocab['</s>']
        for i in range(len(code_ids)):
            code_ids[i].extend( [filler_id] * (longest - len(code_ids[i])) )

        # print(code_ids)
        # print(np.array(code_ids))

        with self._sess.graph.as_default():
            state = self._sess.run(self.model.reset_state)
            feed_dict = {
                self.model.inputd: np.array(code_ids),
                self.model.keep_probability: 1.0
            }
            if self.model.gru:
                for i, h in enumerate(self.model.reset_state):
                    feed_dict[h] = state[i]
            else:
                for i, (c, h) in enumerate(self.model.reset_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h
            
            bpe_token_representation = self._sess.run([self.model.outputs], feed_dict)
            print(bpe_token_representation)
            print(bpe_token_representation[0], bpe_token_representation[0].shape)
            print()
            # print(bpe_token_representation[0][0])
            # print(bpe_token_representation[0][0].shape)
            
            if self.merge == 'sum':
                summed_representations = []
                for representation, sub_sizes in zip(bpe_token_representation[0], subword_sizes):
                    print(representation)
                    summed_representations.append([])
                    index = 0
                    for sub_size in sub_sizes:
                        summed_representations[-1].extend(np.sum(representation[index: index + sub_size], axis=0))
                        index += sub_size
                    # sum()
                return np.array(summed_representations)
            elif self.merge == 'avg':
                averaged_representations = []
                for representation, sub_sizes in zip(bpe_token_representation[0], subword_sizes):
                    print(representation)
                    averaged_representations.append([])
                    index = 0
                    for sub_size in sub_sizes:
                        print('0: ', np.mean(representation[index: index + sub_size], axis=0))
                        print('1: ', np.mean(representation[index: index + sub_size], axis=1))
                        averaged_representations[-1].extend(np.mean(representation[index: index + sub_size], axis=1))
                        index += sub_size
                return averaged_representations
            return bpe_token_representation[0]
    

    def get_embedding_dims(self):
        """[summary]
        
        Returns:
            [type] -- [description]
        """
        return len(self.model.hidden_size)


    def isOOV(self, word):
        """[summary]
        
        Arguments:
            word {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        return False


class Config(object):
  """Configuration"""

  def __init__(self, inits, lr, mgrad, nlayers, nsteps, hsize, mepoch, kp, 
                decay, bsize, tbsize, vsize, gru = True):
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
    self.gru = gru
