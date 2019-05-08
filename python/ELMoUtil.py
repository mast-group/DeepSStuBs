import os
import sys
import tensorflow as tf
import Util

from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings
from BugDetection import parse_data_paths


def create_ELMo_vocabulary(training_data_paths, validation_data_paths):
    vocab = set(['<S>', '</S>'])
    for code_piece in Util.DataReader(training_data_paths, False):
        for token in code_piece['tokens']:
            vocab.add(token.replace(' ', 'U+0020'))
    for code_piece in Util.DataReader(validation_data_paths, False):
        for token in code_piece['tokens']:
            vocab.add(token.replace(' ', 'U+0020'))
    
    # Calculate maximum query size
    max_query = 0
    for code_piece in Util.DataReader(training_data_paths, False):
        if len(code_piece['tokens']) > max_query:
            max_query = len(code_piece['tokens'])
    for code_piece in Util.DataReader(validation_data_paths, False):
        if len(code_piece['tokens']) > max_query:
            max_query = len(code_piece['tokens'])
    print('Maximum query:', max_query)
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

    # Create ELMo Token operation
    elmo_token_op, code_token_ids = create_token_ELMo(options_file, weight_file, \
        token_embedding_file, True)

    with tf.Session() as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())

        tokenized_context = [['ID:func', '(', ')', ';']]
        context_ids = batcher.batch_sentences(tokenized_context)
        print(context_ids)

        # Compute ELMo representations.
        elmo_represenations_ = sess.run(
            elmo_token_op['weighted_op'],
            feed_dict={code_token_ids: context_ids}
        )
        print(elmo_represenations_)

        tokenized_context = [
            ['ID:func', '(', ')', ';'],
            ['ID:func', '(', ')', ';'],
            ['ID:func', '(', ')', ';'],
            ['ID:func', '(', ')', ';'],
            ['ID:func', '(', ')', ';'],
            ['ID:func', '(', ')', ';']
        ]
        context_ids = batcher.batch_sentences(tokenized_context)
        print(context_ids)

        # Compute ELMo representations.
        elmo_represenations_ = sess.run(
            elmo_token_op['weighted_op'],
            feed_dict={code_token_ids: context_ids}
        )
        print(elmo_represenations_)

