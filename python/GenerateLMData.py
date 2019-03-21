'''
Created on March 21, 2019

@author: Rafael-Michael Karampatsis
'''

import sys

from EmbeddingLearnerWord2Vec import EncodedSequenceReader
from os import getcwd
from os.path import join

def instance_tokens_generator(tokens_files):
    data_paths = list(map(lambda f: join(getcwd(), f), tokens_files))
    for token_seq in EncodedSequenceReader(data_paths):
        yield token_seq
        break

def main(args):
    pass

if __name__ == '__main__':
    tokens_files = sys.argv[1]
    for token_seq in instance_tokens_generator(tokens_files):
        print(token_seq)
