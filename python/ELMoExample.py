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

        elmo.get_sequence_embeddings(raw_code)