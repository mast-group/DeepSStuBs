'''
Created on March 21, 2019

@author: Rafael-Michael Karampatsis
'''

import sys

from EmbeddingLearnerWord2Vec import EncodedSequenceReader
from os import getcwd
from os.path import join

def instance_tokens_generator(tokens_files, keep_types=False):
    data_paths = list(map(lambda f: join(getcwd(), f), tokens_files))
    if len(data_paths) is 0:
        print("Must pass token*.json files and at least one data file")
        sys.exit(1)
    
    for token_seq in EncodedSequenceReader(data_paths):
        token_seq = [token.replace(' ', 'U+0020') for token in token_seq]
        if not keep_types:
            token_seq = [token[token.find(':') + 1: ] for token in token_seq]
            yield token_seq
        else: 
            yield token_seq
        # break

def main(args):
    export_file = args[1]
    tokens_files = args[2:]
    with open(export_file, 'w') as f:
        for token_seq in instance_tokens_generator(tokens_files):
            f.write(' '.join(token_seq))
            f.write('\n')

if __name__ == '__main__':
    main(sys.argv)

