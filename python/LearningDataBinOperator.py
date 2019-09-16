'''
Created on Nov 9, 2017

@author: Michael Pradel
'''

import Util
import numpy as np
from collections import Counter
import random

from ELMoClient import *
from ELMoUtil import ELMoMode
from Util import clean_string

type_embedding_size = 5
node_type_embedding_size = 8 # if changing here, then also change in LearningDataBinOperator

class CodePiece(object):
    def __init__(self, left, right, op, src):
        self.left = left
        self.right = right
        self.op = op
        self.src = src
    
    def to_message(self):
        return str(self.src) + " | " + str(self.left) + " | " + str(self.op) + " | " + str(self.right)
    
class LearningData(object):
    def __init__(self):
        self.all_operators = None
        self.stats = {}
        
    def resetStats(self):
        self.stats = {}

    def pre_scan(self, training_data_paths, validation_data_paths):
        all_operators_set = set()
        for bin_op in Util.DataReader(training_data_paths, False):
            all_operators_set.add(bin_op["op"])
        for bin_op in Util.DataReader(validation_data_paths, False):
            all_operators_set.add(bin_op["op"])
        self.all_operators = list(all_operators_set)
    

    def mutate(self, bin_op):
        mutated_bin_op = dict()
        
        mutated_bin_op["left"] = bin_op["left"]
        mutated_bin_op["right"] = bin_op["right"]
        mutated_bin_op["leftType"] = bin_op["leftType"]
        mutated_bin_op["rightType"] = bin_op["rightType"]
        mutated_bin_op["parent"] = bin_op["parent"]
        mutated_bin_op["grandParent"] = bin_op["grandParent"]
        mutated_bin_op["src"] = bin_op["src"]
        mutated_bin_op["opPosition"] = bin_op["opPosition"]
        mutated_tokens = bin_op["tokens"].copy()

        other_operator = bin_op["op"]
        while other_operator == bin_op["op"]:
            other_operator = random.choice(self.all_operators)
        
        mutated_bin_op["op"] = other_operator
        mutated_tokens[mutated_bin_op["opPosition"]] = other_operator

        return mutated_bin_op
    

    def code_features(self, bin_op, embeddings_model, emb_model_type, type_to_vector, node_type_to_vector, code_pieces=None):
        if isinstance(bin_op, list):
            if emb_model_type == 'w2v' or emb_model_type == 'FastText':
                feats = []
                for bin_op_inst in bin_op:
                    x = self.code_features(bin_op_inst, embeddings_model, \
                        emb_model_type, type_to_vector, node_type_to_vector)
                    feats.append(x)
                return feats
            elif emb_model_type == 'ELMo':
                if isinstance(bin_op, list):
                    feats = []
                    queries = []
                    extra_vecs = []
                    for bin_op_inst in bin_op:
                        extra_vecs.append(self._extra_feats(bin_op_inst, type_to_vector, node_type_to_vector))
                        query  = self._to_ELMo_heuristic_query(bin_op_inst, embeddings_model)
                        queries.append(query)
                    
                    embeds = embeddings_model.get_sequence_embeddings(queries)
                    return embeds, np.array(extra_vecs)
                    # for i in range(len(embeds)):
                    #     vec = list(embeds[i].ravel())
                    #     feats.append(vec + extra_vecs[i])
                    # return feats
                else:
                    query  = self._to_ELMo_heuristic_query(bin_op, embeddings_model)
                    extra_vec = self._extra_feats(bin_op, type_to_vector, node_type_to_vector)
                    return embeddings_model.get_sequence_embeddings([query]), np.array(extra_vec)
                    # return embeddings_model.get_sequence_embeddings([query])
                    # x = list(embeddings_model.get_sequence_token_embeddings([query]).ravel()) + extra_vec
                    # return x
        elif emb_model_type == 'BPE':
            if isinstance(bin_op, list):
                feats = []
                queries = []
                extra_vecs = []
                for bin_op_inst in bin_op:
                    extra_vecs.append(self._extra_feats(bin_op_inst, type_to_vector, node_type_to_vector))
                    query  = self._to_ELMo_heuristic_query(bin_op_inst, embeddings_model)
                    queries.append(query)
                    
                embeds = embeddings_model.get_sequence_embeddings(queries)
                for i in range(len(embeds)):
                    vec = list(embeds[i].ravel())
                    feats.append(vec + extra_vecs[i])

                return feats
            else:
                query  = self._to_ELMo_heuristic_query(bin_op_inst, embeddings_model)
                extra_vec = self._extra_feats(bin_op, type_to_vector, node_type_to_vector)
                x = list(embeddings_model.get_sequence_embeddings([query]).ravel()) + extra_vec
                return x
        else:
            left = bin_op["left"]
            right = bin_op["right"]

            if emb_model_type == 'w2v' or emb_model_type == 'FastText':
                operator = bin_op["op"]
                left_type = bin_op["leftType"]
                right_type = bin_op["rightType"]
                parent = bin_op["parent"]
                grand_parent = bin_op["grandParent"]
                src = bin_op["src"]
                
                left_vector = embeddings_model.get_embedding(left)
                right_vector = embeddings_model.get_embedding(right)
            else:
                return None
        
            operator_vector = [0] * len(self.all_operators)
            operator_vector[self.all_operators.index(operator)] = 1
            left_type_vector = type_to_vector.get(left_type, [0]*type_embedding_size)
            right_type_vector = type_to_vector.get(right_type, [0]*type_embedding_size)
            parent_vector = node_type_to_vector.get(parent, [0] * node_type_embedding_size)
            grand_parent_vector = node_type_to_vector.get(grand_parent, [0] * node_type_embedding_size)
            
            x = left_vector + right_vector + operator_vector + left_type_vector + \
                right_type_vector + parent_vector + grand_parent_vector
            return x
        
        if code_pieces != None:
            code_pieces.append(CodePiece(right, left, operator, src))
        
        
    def _extra_feats(self, bin_op, type_to_vector, node_type_to_vector):
        operator = bin_op["op"]
        left_type = bin_op["leftType"]
        right_type = bin_op["rightType"]
        parent = bin_op["parent"]
        grand_parent = bin_op["grandParent"]
        
        operator_vector = [0] * len(self.all_operators)
        operator_vector[self.all_operators.index(operator)] = 1
        left_type_vector = type_to_vector.get(left_type, [0]*type_embedding_size)
        right_type_vector = type_to_vector.get(right_type, [0]*type_embedding_size)
        parent_vector = node_type_to_vector.get(parent, [0] * node_type_embedding_size)
        grand_parent_vector = node_type_to_vector.get(grand_parent, [0] * node_type_embedding_size)
        
        return operator_vector + left_type_vector + right_type_vector + parent_vector + grand_parent_vector
    

    def _to_ELMo_heuristic_query(self, bin_op, embeddings_model):
        left = bin_op["left"]
        right = bin_op["right"]
        operator = bin_op["op"]
        query = '%s %s %s' % (clean_string(left), clean_string(operator), clean_string(right))

        return query.split()


    def code_to_xy_FastText_pairs(self, bin_op, xs, ys, name_to_vector, type_to_vector, node_type_to_vector, code_pieces=None):
        left = bin_op["left"]
        right = bin_op["right"]
        operator = bin_op["op"]
        left_type = bin_op["leftType"]
        right_type = bin_op["rightType"]
        parent = bin_op["parent"]
        grand_parent = bin_op["grandParent"]
        src = bin_op["src"]
        if not (left in name_to_vector):
            left = 'UNK'
            # return
        if not (right in name_to_vector):
            right = 'UNK'
            # return
    
        left_vector = list(name_to_vector[left])
        right_vector = list(name_to_vector[right])
        operator_vector = [0] * len(self.all_operators)
        operator_vector[self.all_operators.index(operator)] = 1
        left_type_vector = type_to_vector.get(left_type, [0]*type_embedding_size)
        right_type_vector = type_to_vector.get(right_type, [0]*type_embedding_size)
        parent_vector = node_type_to_vector[parent]
        grand_parent_vector = node_type_to_vector[grand_parent]
        
        # for all xy-pairs: y value = probability that incorrect
        x_correct = left_vector + right_vector + operator_vector
        x_correct += left_type_vector + right_type_vector + parent_vector + grand_parent_vector
        y_correct = [0]
        xs.append(x_correct)
        ys.append(y_correct)
        if code_pieces != None:
            code_pieces.append(CodePiece(left, right, operator, src))
        
        # pick some other, likely incorrect operator
        other_operator_vector = None
        while other_operator_vector == None:
            other_operator = random.choice(self.all_operators)
            if other_operator != operator:
                other_operator_vector = [0] * len(self.all_operators)
                other_operator_vector[self.all_operators.index(other_operator)] = 1
        
        x_incorrect = left_vector + right_vector + other_operator_vector + left_type_vector + right_type_vector + parent_vector + grand_parent_vector
        y_incorrect = [1]
        xs.append(x_incorrect)
        ys.append(y_incorrect)
        if code_pieces != None:
            code_pieces.append(CodePiece(left, right, other_operator, src))


    def code_to_xy_pairs(self, bin_op, xs, ys, name_to_vector, type_to_vector, node_type_to_vector, code_pieces=None):
        left = bin_op["left"]
        right = bin_op["right"]
        operator = bin_op["op"]
        left_type = bin_op["leftType"]
        right_type = bin_op["rightType"]
        parent = bin_op["parent"]
        grand_parent = bin_op["grandParent"]
        src = bin_op["src"]
        if not (left in name_to_vector):
            left = 'UNK'
            # return
        if not (right in name_to_vector):
            right = 'UNK'
            # return
    
        left_vector = name_to_vector[left]
        right_vector = name_to_vector[right]
        operator_vector = [0] * len(self.all_operators)
        operator_vector[self.all_operators.index(operator)] = 1
        left_type_vector = type_to_vector.get(left_type, [0]*type_embedding_size)
        right_type_vector = type_to_vector.get(right_type, [0]*type_embedding_size)
        parent_vector = node_type_to_vector.get(parent, [0] * node_type_embedding_size)
        grand_parent_vector = node_type_to_vector.get(grand_parent, [0] * node_type_embedding_size)
        
        # for all xy-pairs: y value = probability that incorrect
        x_correct = left_vector + right_vector + operator_vector + left_type_vector + right_type_vector + parent_vector + grand_parent_vector
        y_correct = [0]
        xs.append(x_correct)
        ys.append(y_correct)
        if code_pieces != None:
            code_pieces.append(CodePiece(left, right, operator, src))
        
        # pick some other, likely incorrect operator
        other_operator_vector = None
        while other_operator_vector == None:
            other_operator = random.choice(self.all_operators)
            if other_operator != operator:
                other_operator_vector = [0] * len(self.all_operators)
                other_operator_vector[self.all_operators.index(other_operator)] = 1
        
        x_incorrect = left_vector + right_vector + other_operator_vector + left_type_vector + right_type_vector + parent_vector + grand_parent_vector
        y_incorrect = [1]
        xs.append(x_incorrect)
        ys.append(y_incorrect)
        if code_pieces != None:
            code_pieces.append(CodePiece(left, right, other_operator, src))

    def code_to_ELMo_xy_pairs(self, bin_ops, xs, ys, name_to_vector, type_to_vector, node_type_to_vector, ELMoModel, ops=None):        
        queries = []
        type_representations = []
        for bin_op in bin_ops:
            left = bin_op["left"]
            right = bin_op["right"]
            operator = bin_op["op"]
            left_type = bin_op["leftType"]
            right_type = bin_op["rightType"]
            parent = bin_op["parent"]
            grand_parent = bin_op["grandParent"]
            src = bin_op["src"]
            tokens = bin_op["tokens"]
            op_position = int(bin_op["opPosition"])

            queries.append(tokens)
            other_operator = None
            while other_operator == None or other_operator == operator:
                other_operator = random.choice(self.all_operators)
                # print(operator, other_operator)
            wrong_query = tokens.copy()
            wrong_query[op_position] = "STD:" + other_operator
            queries.append(wrong_query)

            left_type_vector = type_to_vector.get(left_type, [0]*type_embedding_size)
            right_type_vector = type_to_vector.get(right_type, [0]*type_embedding_size)
            parent_vector = node_type_to_vector[parent]
            grand_parent_vector = node_type_to_vector[grand_parent]

            type_representations.append(left_type_vector + right_type_vector)
            type_representations.append(right_type_vector + left_type_vector)
        type_representations.append(None)

        queries.append([""] * 30)
        
        elmo_representations = ELMoModel.query(queries, ELMoMode.ALL)
        for i, features in enumerate(zip(elmo_representations, type_representations)):
            representation, type_feats = features
            if i == len(elmo_representations) - 1: break
            # xs.append(np.append(representation, np.array(type_feats)))
            xs.append(representation)
            ys.append([i % 2])
    

    def code_to_ELMo_baseline_xy_pairs(self, bin_ops, xs, ys, name_to_vector, type_to_vector, node_type_to_vector, ELMoModel, ops=None):
        queries = []
        type_representations = []
        for bin_op in bin_ops:
            left = bin_op["left"]
            right = bin_op["right"]
            operator = bin_op["op"]
            left_type = bin_op["leftType"]
            right_type = bin_op["rightType"]
            parent = bin_op["parent"]
            grand_parent = bin_op["grandParent"]
            src = bin_op["src"]
            tokens = bin_op["tokens"]
            op_position = int(bin_op["opPosition"])

            other_operator = None
            while other_operator == None or other_operator == operator:
                other_operator = random.choice(self.all_operators)
            
            correct_code = '%s %s %s' % (clean_string(left), operator, clean_string(right))
            buggy_code = '%s %s %s' % (clean_string(left), other_operator, clean_string(right))
            queries.append( correct_code.split(' ') )
            queries.append( buggy_code.split(' ') )
        
        if len(queries) == 0:
            return
        elmo_representations = ELMoModel.query(queries, ELMoMode.ALL)
        for i, features in enumerate(elmo_representations):#zip(elmo_representations, type_representations)):
            # representation, type_feats = features
            # if i == len(elmo_representations) - 1: break
            # xs.append(np.append(representation, np.array(type_feats)))
            xs.append(features)
            ys.append([i % 2])


    def anomaly_score(self, y_prediction_orig, y_prediction_changed):
        return y_prediction_orig
    
    def normal_score(self, y_prediction_orig, y_prediction_changed):
        return y_prediction_changed
    