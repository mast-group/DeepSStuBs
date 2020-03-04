import time
import sys
import json
import os
from os.path import join
from os import getcwd
from collections import Counter, namedtuple
import math

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)

import model.ModelFactory
from model.ModelFactory import *
import model.ELMoModel
from ELMoModel import *

from gensim.models import FastText
from threading import Thread
from Util import *
from bilm import TokenBatcher




if __name__ == '__main__':
    raw_code = [
        'STD:function STD:( ID:e STD:, ID:tags STD:) STD:{ ID:tags STD:. ID:should STD:. ' + 
        'ID:have STD:. ID:lengthOf STD:( LIT:1 STD:) STD:; ID:done STD:( STD:) STD:; STD:} '
        'STD:) STD:; STD:} STD:) STD:; STD:} STD:) STD:; STD:} STD:) STD:; STD:} STD:) STD:; ' +
        'STD:} STD:) STD:; STD:} STD:) STD:; STD:} STD:) STD:; STD:} STD:) STD:; STD:} STD:) ' +
        'STD:; STD:} STD:) STD:;',
        'STD:var ID:gfm STD:= ID:require STD:( LIT:github-flavored-markdown STD:) STD:;'
    ]
    tokenized_code = [sentence.split() for sentence in raw_code]
    print(tokenized_code)

    vocab_file = 'ELMoVocab.txt'
    # Location of pretrained LM. .
    model_dir = '/disk/scratch/mpatsis/eddie/models/phog/js/elmo/emb100_hidden1024_steps20_drop0.1/'
    options_file = os.path.join(model_dir, 'query_options.json')
    weight_file = os.path.join(model_dir, 'weights/weights.hdf5')

    with tf.Session() as sess:
        elmo = ELMoModel(model_dir, vocab_file, weight_file, options_file, sess)

        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())

        time_start = time.time()

        token_emb = elmo.get_embedding('STD:function')
        print(token_emb)
        print(token_emb[0].shape)
        
        token_embs = elmo.get_sequence_token_embeddings(tokenized_code)
        print(token_embs)
        print(token_embs[0].shape)

        time_end = time.time()
        print('Lasted: ', time_end - time_start)

