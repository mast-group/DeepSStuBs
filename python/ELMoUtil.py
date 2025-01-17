import os
import sys
import numpy as np
import tensorflow as tf
import Util

from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings

from enum import Enum, unique


def create_ELMo_vocabulary(training_data_paths, validation_data_paths):
    vocab = set(['<S>', '</S>'])
    for code_piece in Util.DataReader(training_data_paths, False):
        for token in code_piece['tokens']:
            vocab.add(token.replace(' ', 'U+0020'))
    for code_piece in Util.DataReader(validation_data_paths, False):
        for token in code_piece['tokens']:
            vocab.add(token.replace(' ', 'U+0020'))
    
    return vocab


def save_vocab(vocab_file):
    with open(vocab_file, 'w') as fout:
        fout.write('\n'.join(vocab))


def save_token_embeddings(vocab_file, options_file, weight_file, token_embedding_file):
    dump_token_embeddings(vocab_file, options_file, weight_file, token_embedding_file)
    tf.reset_default_graph()


def create_token_ELMo(options_file, weight_file, token_embedding_file, use_top_only=True):
    # Input placeholders to the biLM.
    code_token_ids = tf.placeholder('int32', shape=(None, None))

    # Build the biLM graph.
    bilm = BidirectionalLanguageModel(
        options_file,
        weight_file,
        use_character_inputs=False,
        embedding_weight_file=token_embedding_file
    )

    code_embeddings_op = bilm(code_token_ids)
    elmo_token_op = weight_layers('ELMo', code_embeddings_op, l2_coef=0.0, use_top_only=use_top_only)
    return elmo_token_op, code_token_ids


def parse_data_paths(args):
    training_data_paths = []
    eval_data_paths = []
    mode = None
    for arg in args:
        if arg == "--trainingData":
            assert mode == None
            mode = "trainingData"
        elif arg == "--validationData":
            assert mode == "trainingData"
            mode = "validationData"
        else:
            path = join(getcwd(), arg)
            if mode == "trainingData":
                training_data_paths.append(path)
            elif mode == "validationData":
                eval_data_paths.append(path)
            else:
                print("Incorrect arguments")
                sys.exit(0)
    return [training_data_paths, eval_data_paths]


@unique
class ELMoMode(Enum):
    ALL = 1
    CENTROID = 2
    SUM = 3

class ELMoModel(object):
    def __init__(self, sess, batcher, elmo_token_op, code_token_ids, threshold=30, dims=200):
        self.sess = sess
        self.batcher = batcher
        self.elmo_token_op = elmo_token_op
        self.code_token_ids = code_token_ids
        self.threshold = threshold
        self.dims = dims
    
    def query(self, queries, mode=ELMoMode.ALL):
        context_ids = self.batcher.batch_sentences(queries)
        # print(context_ids)
        
        # Compute ELMo representations.
        elmo_represenations_ = self.sess.run(
            self.elmo_token_op['weighted_op'],
            feed_dict={self.code_token_ids: context_ids}
        )
        # print(elmo_represenations_)

        if mode == ELMoMode.CENTROID:
            elmo_represenations = []
            for i, query in enumerate(queries):
                elmo_represenation = elmo_represenations_[i]
                tokens = len(query)
                # print(elmo_represenation[: tokens], tokens)
                elmo_represenation = elmo_represenation[: tokens]
                elmo_represenation = np.mean(elmo_represenation, axis=0)
                # print(elmo_represenation)
                elmo_represenations.append(elmo_represenation)
            return np.array(elmo_represenations).reshape(len(context_ids), -1)
            # return np.mean(elmo_represenations_, axis=1)
        elif mode == ELMoMode.SUM:
            return np.sum(elmo_represenations_, axis=1)
        # print('all case')
        return elmo_represenations_.reshape(len(context_ids), -1)


if __name__ == '__main__':
    training_data_paths, validation_data_paths = parse_data_paths(sys.argv[1:])
    vocab = create_ELMo_vocabulary(training_data_paths, validation_data_paths)
    vocab_file = 'ELMoVocab.txt'
    # save_vocab('ELMoVocab.txt')

    # Location of pretrained LM.  Here we use the test fixtures.
    model_dir = '/disk/scratch/mpatsis/eddie/models/phog/js/elmo/emb100_hidden1024_steps20_drop0.1/'
    options_file = os.path.join(model_dir, 'query_options.json')
    weight_file = os.path.join(model_dir, 'weights/weights.hdf5')
    
    # Dump the token embeddings to a file. Run this once for your dataset.
    token_embedding_file = 'elmo_token_embeddings.hdf5'
    # save_token_embeddings(vocab_file, options_file, weight_file, token_embedding_file)

    # Create a TokenBatcher to map text to token ids.
    batcher = TokenBatcher(vocab_file)

    USE_ELMO_TOP_ONLY = True
    # Create ELMo Token operation
    elmo_token_op, code_token_ids = create_token_ELMo(options_file, weight_file, \
        token_embedding_file, USE_ELMO_TOP_ONLY)
    

    with tf.Session() as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())
        ELMoModel = ELMoModel(sess, batcher, elmo_token_op, code_token_ids)

        tokenized_context = [['ID:func', 'STD:(', 'STD:)', 'STD:;']]
        context_ids = batcher.batch_sentences(tokenized_context)
        print(context_ids)

        # Compute ELMo representations.
        elmo_represenations_ = ELMoModel.query(tokenized_context, ELMoMode.CENTROID)
        # elmo_represenations_ = sess.run(
        #     elmo_token_op['weighted_op'],
        #     feed_dict={code_token_ids: context_ids}
        # )
        print(elmo_represenations_)
        print(elmo_represenations_.shape)

        # print(np.squeeze(elmo_represenations_.reshape(len(context_ids), -1), 0))
        # print(np.squeeze(elmo_represenations_.reshape(len(context_ids), -1), 0).shape)
        # ar = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,])
        # print(np.append(elmo_represenations_.reshape(len(context_ids), -1), ar))

        # print(np.mean(elmo_represenations_, axis=1))
        # print(np.mean(elmo_represenations_, axis=1).shape)

        # tokenized_context = [
        #     ['ID:func', 'STD:(', 'STD:)', 'STD:;'],
        #     ['ID:func', 'STD:(', 'STD:)', 'STD:;'],
        #     ['ID:func', 'STD:(', 'STD:)', 'STD:;'],
        #     ['ID:func', 'STD:(', 'STD:)', 'STD:;'],
        #     ['ID:func', 'STD:(', 'STD:)', 'STD:;'],
        #     ['ID:func', 'STD:(', 'STD:)', 'STD:;']
        # ]
        # context_ids = batcher.batch_sentences(tokenized_context)
        # print(context_ids, len(context_ids))

        # # Compute ELMo representations.
        # elmo_represenations_ = sess.run(
        #     elmo_token_op['weighted_op'],
        #     feed_dict={code_token_ids: context_ids}
        # )
        # print(elmo_represenations_)
        # print(np.mean(elmo_represenations_, axis=1).shape)


        # tokenized_context = [
        #     ['ID:func', 'STD:(', 'STD:)', 'STD:;', 'ID:func', 'STD:(', 'STD:)', 'STD:;'],
        #     ['ID:func', 'STD:(', 'STD:)', 'STD:;']
        # ]
        # context_ids = batcher.batch_sentences(tokenized_context)
        # print(context_ids)

        # # Compute ELMo representations.
        # elmo_represenations_ = sess.run(
        #     elmo_token_op['weighted_op'],
        #     feed_dict={code_token_ids: context_ids}
        # )
        # print(elmo_represenations_)

        # for i in range(100):
        #     tokenized_context = [['ID:func', 'STD:(', 'STD:)', 'STD:;'] * 8] * 128
        #     context_ids = batcher.batch_sentences(tokenized_context)
        #     print(context_ids)
        #     # Compute ELMo representations.
        #     elmo_represenations_ = sess.run(
        #         elmo_token_op['weighted_op'],
        #         feed_dict={code_token_ids: context_ids}
        #     )
        #     print(len(elmo_represenations_))
