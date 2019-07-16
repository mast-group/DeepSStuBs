'''
Created on Nov 13, 2017

@author: Michael Pradel
'''

import Util
from collections import namedtuple
import random

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

Operand = namedtuple('Operand', ['op', 'type'])
    
class LearningData(object):
    def __init__(self):
        self.file_to_operands = dict() # string to set of Operands
        self.stats = {}

    def resetStats(self):
        self.stats = {}

    def pre_scan(self, training_data_paths, validation_data_paths):
        all_operators_set = set()
        for bin_op in Util.DataReader(training_data_paths, False):
            file = bin_op["src"].split(" : ")[0]
            operands = self.file_to_operands.setdefault(file, dict())
            # operands = self.file_to_operands.setdefault(file, set())
            left_operand = Operand(bin_op["left"], bin_op["leftType"])
            right_operand = Operand(bin_op["right"], bin_op["rightType"])
            if not left_operand in operands: operands[left_operand] = bin_op["tokens"][:bin_op["opPosition"]]
            if not right_operand in operands: operands[right_operand] = bin_op["tokens"][bin_op["opPosition"] + 1: ]
            # operands.add(left_operand)
            # operands.add(right_operand)
            
            all_operators_set.add(bin_op["op"])
        for bin_op in Util.DataReader(validation_data_paths, False):
            file = bin_op["src"].split(" : ")[0]
            operands = self.file_to_operands.setdefault(file, dict())
            # operands = self.file_to_operands.setdefault(file, set())
            left_operand = Operand(bin_op["left"], bin_op["leftType"])
            right_operand = Operand(bin_op["right"], bin_op["rightType"])
            if not left_operand in operands: operands[left_operand] = bin_op["tokens"][:bin_op["opPosition"]]
            if not right_operand in operands: operands[right_operand] = bin_op["tokens"][bin_op["opPosition"] + 1: ]
            # operands.add(left_operand)
            # operands.add(right_operand)
            
            all_operators_set.add(bin_op["op"])
        self.all_operators = list(all_operators_set)
    
    
    def mutate(self, bin_op):
        mutated_bin_op = dict()
        
        mutated_bin_op["left"] = bin_op["left"]
        mutated_bin_op["right"] = bin_op["right"]
        mutated_bin_op["op"] = bin_op["op"]
        mutated_bin_op["left_type"] = bin_op["leftType"]
        mutated_bin_op["right_type"] = bin_op["rightType"]
        mutated_bin_op["parent"] = bin_op["parent"]
        mutated_bin_op["grand_parent"] = bin_op["grandParent"]
        mutated_bin_op["src"] = bin_op["src"]
        mutated_bin_op["opPosition"] = bin_op["opPosition"]
        mutated_tokens = bin_op["tokens"].copy()
        
        # find an alternative operand in the same file
        replace_left = random.random() < 0.5
        if replace_left:
            to_replace_operand = mutated_bin_op["left"]
        else:
            to_replace_operand = mutated_bin_op["right"]
        file = bin_op["src"].split(" : ")[0]
        all_operands = self.file_to_operands[file].keys()
        tries_left = 100
        found = False
        if len(all_operands) == 1:
            return;

        while (not found) and tries_left > 0:
            other_operand = random.choice(list(all_operands))
            # This if here is problematic if it is inh word2vec, because it only keeps operands in vocab
            # if other_operand.op in name_to_vector and other_operand.op != to_replace_operand:
            if other_operand.op != to_replace_operand:
                found = True
            tries_left -= 1
            
        if not found:
            # print('Did not find operand')
            return
        
        if replace_left:
            mutated_bin_op["left"] = other_operand.op
            mutated_bin_op["left_type"] = other_operand.type
            mutated_tokens = self.file_to_operands[file][other_operand] + bin_op["tokens"][bin_op["opPosition"]:]
        else:
            mutated_bin_op["right"] = other_operand.op
            mutated_bin_op["right_type"] = other_operand.type
            mutated_tokens = bin_op["tokens"][: bin_op["opPosition"] + 1] + self.file_to_operands[file][other_operand]

        return mutated_bin_op
    

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
        
        # find an alternative operand in the same file
        replace_left = random.random() < 0.5
        if replace_left:
            to_replace_operand = left
        else:
            to_replace_operand = right
        file = src.split(" : ")[0]
        all_operands = self.file_to_operands[file]
        tries_left = 100
        found = False
        while (not found) and tries_left > 0:
            other_operand = random.choice(list(all_operands))
            if other_operand.op in name_to_vector and other_operand.op != to_replace_operand:
                found = True
            tries_left -= 1
            
        if not found:
            return
        
        # for all xy-pairs: y value = probability that incorrect
        x_correct = left_vector + right_vector + operator_vector + left_type_vector + right_type_vector + parent_vector + grand_parent_vector
        y_correct = [0]
        xs.append(x_correct)
        ys.append(y_correct)
        if code_pieces != None:
            code_pieces.append(CodePiece(left, right, operator, src))
        
        other_operand_vector = list(name_to_vector[other_operand.op])
        other_operand_type_vector = type_to_vector[other_operand.type]
        # replace one operand with the alternative one
        if replace_left:
            x_incorrect = other_operand_vector + right_vector + operator_vector + other_operand_type_vector + right_type_vector + parent_vector + grand_parent_vector
        else:
            x_incorrect = left_vector + other_operand_vector + operator_vector + right_type_vector + other_operand_type_vector + parent_vector + grand_parent_vector
        y_incorrect = [1]
        xs.append(x_incorrect)
        ys.append(y_incorrect)
        if code_pieces != None:
            code_pieces.append(CodePiece(right, left, operator, src))

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
        
        # find an alternative operand in the same file
        replace_left = random.random() < 0.5
        if replace_left:
            to_replace_operand = left
        else:
            to_replace_operand = right
        file = src.split(" : ")[0]
        all_operands = self.file_to_operands[file]
        tries_left = 100
        found = False
        while (not found) and tries_left > 0:
            other_operand = random.choice(list(all_operands))
            if other_operand.op in name_to_vector and other_operand.op != to_replace_operand:
                found = True
            tries_left -= 1
            
        if not found:
            return
        
        # for all xy-pairs: y value = probability that incorrect
        x_correct = left_vector + right_vector + operator_vector + left_type_vector + right_type_vector + parent_vector + grand_parent_vector
        y_correct = [0]
        xs.append(x_correct)
        ys.append(y_correct)
        if code_pieces != None:
            code_pieces.append(CodePiece(left, right, operator, src))
        
        other_operand_vector = name_to_vector[other_operand.op]
        other_operand_type_vector = type_to_vector[other_operand.type]
        # replace one operand with the alternative one
        if replace_left:
            x_incorrect = other_operand_vector + right_vector + operator_vector + other_operand_type_vector + right_type_vector + parent_vector + grand_parent_vector
        else:
            x_incorrect = left_vector + other_operand_vector + operator_vector + right_type_vector + other_operand_type_vector + parent_vector + grand_parent_vector
        y_incorrect = [1]
        xs.append(x_incorrect)
        ys.append(y_incorrect)
        if code_pieces != None:
            code_pieces.append(CodePiece(right, left, operator, src))
        
    def anomaly_score(self, y_prediction_orig, y_prediction_changed):
        return y_prediction_orig
    
    def normal_score(self, y_prediction_orig, y_prediction_changed):
        return y_prediction_changed
            