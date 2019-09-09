'''
Created on March 21, 2019

@author: Rafael-Michael Karampatsis
'''

import sys

from EmbeddingLearnerWord2Vec import EncodedSequenceReader
from os import getcwd
from os.path import join
from Util import clean_string


def instance_tokens_generator(tokens_files, keep_types=True):
    sequences = 0
    data_paths = list(map(lambda f: join(getcwd(), f), tokens_files))
    if len(data_paths) is 0:
        print("Must pass token*.json files and at least one data file")
        sys.exit(1)
    
    for token_seq in EncodedSequenceReader(data_paths):
        sequences += 1
        token_seq = [clean_string(token) for token in token_seq]
        
        if not keep_types:
            token_seq = [token[token.find(':') + 1: ] for token in token_seq]
            yield token_seq
        else: 
            yield token_seq
    print('Sequences: %d' % sequences)
        # break

def main(args):
    keep_types = args[1]
    export_file = args[2]
    tokens_files = args[3:]
    with open(export_file, 'w') as f:
        yields = 0
        for token_seq in instance_tokens_generator(tokens_files, False):
            yields += 1
            f.write(' '.join(token_seq))
            f.write('\n')
    print('yields: %d' % yields)
    

if __name__ == '__main__':
    main(sys.argv)

