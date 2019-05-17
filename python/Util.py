'''
Created on Oct 26, 2017

@author: Michael Pradel
'''

from scipy.spatial.distance import cosine
import random
import json
import linecache
import sys


def in_group_similarity(vector_group):
    vector_group = list(vector_group)
    in_group_simil = 0.0
    in_group_ctr = 0
    for i in range(0, len(vector_group)):
        vector1 = vector_group[i]
        for j in range(i+1, len(vector_group)):
            vector2 = vector_group[j]
            in_group_simil += (1 - cosine(vector1, vector2))
            in_group_ctr += 1
    in_group_simil = in_group_simil / in_group_ctr
    return in_group_simil

def out_group_similarity(vector_group, other_vectors):
    other_vectors = list(other_vectors)
    out_vectors = []
    for _ in range(20):
        out_vectors.append(other_vectors[random.randint(0, len(other_vectors) - 1)])
    out_group_simil = 0.0
    out_group_ctr = 0
    for vector1 in vector_group:
        for vector2 in out_vectors:
            out_group_simil += (1 - cosine(vector1, vector2))
            out_group_ctr += 1
    out_group_simil = out_group_simil / out_group_ctr
    return out_group_simil

class DataReader(object):
    def __init__(self, data_paths, logging=True):
        self.data_paths = data_paths
        self.logging = logging
         
    def __iter__(self):
        for data_path in self.data_paths:
            if self.logging: print("Reading file " + data_path)
            with open(data_path) as file:
                calls = json.load(file)
                for call in calls:
                    yield call
                    
def analyze_histograms(counter):
    total = sum(counter.values())
    sorted_pairs = counter.most_common()
    percentages_to_cover = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    nb_covered = 0
    pairs_covered = 0
    for pair in sorted_pairs:
        nb_covered += pair[1]
        pairs_covered += 1
        percentage_covered = (nb_covered * 1.0) / total
        done = False
        while not done and len(percentages_to_cover) > 0:
            next_percentage = percentages_to_cover[0]
            if percentage_covered >= next_percentage:
                print(str(pairs_covered) + " most frequent terms cover " + str(next_percentage) + " of all terms") 
                percentages_to_cover = percentages_to_cover[1:]
            else:
                done = True


def mean(values, weights=None):
    """[summary]
    
    Arguments:
        values {[type]} -- [description]
    
    Keyword Arguments:
        weights {[type]} -- [description] (default: {None})
    
    Returns:
        [type] -- [description]
    """
    if weights is None:
        mean = 0.0
        for n, value in enumerate(values, start=1):
            mean = mean + (value - mean) / n
        return mean
    else:
        return weighted_mean(values, weights)


def weighted_mean(values, weights):
    """[summary]
    
    Arguments:
        values {[type]} -- [description]
        weights {[type]} -- [description]
    """
    weight_sum = 0.0
    mean = 0.0
    for n, (value, weight) in enumerate(zip(values, weights), start=1):
        weight_sum += weight
        mean = mean + weight / weight_sum * (value - mean)
    return mean


def print_exception():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


def clean_string(str):
    str = str.replace(' ', 'U+0020')
    str = str.replace('\n', '\\n')
    str = str.replace('\r', '\\r')
    str = str.replace('\t', '\\t')
