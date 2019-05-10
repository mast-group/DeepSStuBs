'''
Created on Nov 9, 2017

@author: Michael Pradel
'''

import Util
from collections import Counter

from ELMoClient import *
import itertools


name_embedding_size = 200
file_name_embedding_size = 50
type_embedding_size = 5

class CodePiece(object):
    def __init__(self, callee, arguments, src):
        self.callee = callee
        self.arguments = arguments
        self.src = src
    
    def to_message(self):
        return str(self.src) + " | " + str(self.callee) + " | " + str(self.arguments)
        
class LearningData(object):
    def is_known_type(self, t):
        return t == "boolean" or t == "number" or t == "object" or t == "regex" or t == "string"
    
    def resetStats(self):
        self.stats = {"calls": 0, "calls_with_two_args": 0, "calls_with_known_names": 0,
                      "calls_with_known_base_object": 0, "calls_with_known_types": 0,
                      "calls_with_both_known_types": 0,
                      "calls_with_known_parameters" :0}
    
    def pre_scan(self, training_data_paths, validation_data_paths):
        print("Stats on training data")
        self.gather_stats(training_data_paths)
        print("Stats on validation data")
        self.gather_stats(validation_data_paths)

    def gather_stats(self, data_paths):
        callee_to_freq = Counter()
        argument_to_freq = Counter()
        
        for call in Util.DataReader(data_paths):
            callee_to_freq[call["callee"]] += 1
            for argument in call["arguments"]:
                argument_to_freq[argument] += 1
            
        print("Unique callees        : " + str(len(callee_to_freq)))
        print("  " + "\n  ".join(str(x) for x in callee_to_freq.most_common(10)))
        Util.analyze_histograms(callee_to_freq)
        print("Unique arguments      : " + str(len(argument_to_freq)))
        print("  " + "\n  ".join(str(x) for x in argument_to_freq.most_common(10)))
        Util.analyze_histograms(argument_to_freq)
        
    def code_to_xy_pairs(self, call, xs, ys, name_to_vector, type_to_vector, node_type_to_vector, calls=None):
        arguments = call["arguments"]
        self.stats["calls"] += 1
        if len(arguments) != 2:
            return
        self.stats["calls_with_two_args"] += 1
        
        # mandatory information: callee and argument names        
        callee_string = call["callee"]
        argument_strings = call["arguments"]
        if not (callee_string in name_to_vector):
           callee_string = 'UNK'
            # return
        for e, argument_string in enumerate(argument_strings):
            if not (argument_string in name_to_vector):
                argument_strings[e] = 'UNK'
                # return
        # print('will calculate feats')
        self.stats["calls_with_known_names"] += 1
        callee_vector = name_to_vector[callee_string]
        argument0_vector = name_to_vector[argument_strings[0]]
        argument1_vector = name_to_vector[argument_strings[1]]

        # optional information: base object, argument types, etc.
        base_string = call["base"]
        base_vector = name_to_vector.get(base_string, [0]*name_embedding_size)
        if base_string in name_to_vector:
            self.stats["calls_with_known_base_object"] += 1
        
        argument_type_strings = call["argumentTypes"]
        argument0_type_vector = type_to_vector.get(argument_type_strings[0], [0]*type_embedding_size)
        argument1_type_vector = type_to_vector.get(argument_type_strings[1], [0]*type_embedding_size)
        if (self.is_known_type(argument_type_strings[0]) or self.is_known_type(argument_type_strings[1])):
            self.stats["calls_with_known_types"] += 1
        if (self.is_known_type(argument_type_strings[0]) and self.is_known_type(argument_type_strings[1])):
            self.stats["calls_with_both_known_types"] += 1
        
        parameter_strings = call["parameters"]
        parameter0_vector = name_to_vector.get(parameter_strings[0], [0]*name_embedding_size)
        parameter1_vector = name_to_vector.get(parameter_strings[1], [0]*name_embedding_size)
        if (parameter_strings[0] in name_to_vector or parameter_strings[1] in name_to_vector):
            self.stats["calls_with_known_parameters"] += 1
        
        # for all xy-pairs: y value = probability that incorrect
        x_keep = callee_vector + argument0_vector + argument1_vector
        x_keep += base_vector + argument0_type_vector + argument1_type_vector
        x_keep += parameter0_vector + parameter1_vector #+ file_name_vector
        y_keep = [0]
        # print("callee_vector: ", callee_vector)
        # print("argument0_vector: ", argument0_vector)
        # print("argument1_vector: ", argument1_vector)
        # print("base_vector: ", base_vector)
        # print("argument0_type_vector: ", argument0_type_vector)
        # print("argument1_type_vector: ", argument1_type_vector)
        # print("parameter0_vector: ", parameter0_vector)
        # print("parameter1_vector: ", parameter1_vector)
        # print("\n\n")
        xs.append(x_keep)
        ys.append(y_keep)
        if calls != None:
            calls.append(CodePiece(callee_string, argument_strings, call["src"]))
        
        x_swap = callee_vector + argument1_vector + argument0_vector
        x_swap += base_vector + argument1_type_vector + argument0_type_vector
        x_swap += parameter0_vector + parameter1_vector #+ file_name_vector
        y_swap = [1]
        xs.append(x_swap)
        ys.append(y_swap)
        if calls != None:
            calls.append(CodePiece(callee_string, argument_strings, call["src"]))

    def code_to_ELMo_xy_pairs(self, func_calls, xs, ys, name_to_vector, type_to_vector, node_type_to_vector, ELMoModel, calls=None):
        queries = []
        for call in func_calls:
            arguments = call["arguments"]
            self.stats["calls"] += 1
            if len(arguments) != 2:
                continue
            self.stats["calls_with_two_args"] += 1
            queries.append(call["tokens"])
            queries.append(call["swappedTokens"])
        queries.append([""] * 30)
        
        elmo_representations = ELMoModel.query(queries)
        for i, representation in enumerate(elmo_representations):
            if i == len(elmo_representations) - 1: break
            xs.append(representation)
            ys.append([i % 2])
        
        # if calls != None:
        #     calls.append(CodePiece(callee_string, argument_strings, call["src"]))

    
    def code_to_ELMo_server_xy_pairs(self, call, xs, ys, name_to_vector, type_to_vector, node_type_to_vector, socket, calls=None):
        arguments = call["arguments"]
        self.stats["calls"] += 1
        if len(arguments) != 2:
            return
        self.stats["calls_with_two_args"] += 1
        
        # mandatory information: callee and argument names        
        callee_string = call["callee"]
        argument_strings = call["arguments"]
        if not (callee_string in name_to_vector):
            return
        for argument_string in argument_strings:
            if not (argument_string in name_to_vector):
                return
        self.stats["calls_with_known_names"] += 1
        callee_vector = name_to_vector[callee_string]
        # argument0_vector = name_to_vector[argument_strings[0]]
        # argument1_vector = name_to_vector[argument_strings[1]]
        
        correct_code = ""
        # print("callee string:" + callee_string)
        if call["base"] != "":
            correct_code += call["base"].replace(' ', 'U+0020') + " . "
        buggy_code = correct_code
        correct_code += " %s ( %s , %s )" % (callee_string, \
            argument_strings[0].replace(' ', 'U+0020'), argument_strings[1].replace(' ', 'U+0020'))
        # print(call["base"] + " " + callee_string + " " + \
        #     argument_strings[0].replace(' ', 'U+0020') + ", " + \
        #         argument_strings[1].replace(' ', 'U+0020'))
        # print(len(correct_code.split()), correct_code)
        buggy_code += " %s ( %s , %s )" % (callee_string, \
            argument_strings[1].replace(' ', 'U+0020'), argument_strings[0].replace(' ', 'U+0020'))
        
        elmo_representations = query([correct_code, buggy_code], socket)
        correct_vectors = elmo_representations[0][0]
        # print('correct vector:', correct_vectors)
        wrong_vectors = elmo_representations[0][1]
        

        # optional information: base object, argument types, etc.
        base_string = call["base"]
        base_vector = name_to_vector.get(base_string, [0]*name_embedding_size)
        if base_string in name_to_vector:
            self.stats["calls_with_known_base_object"] += 1
        
        argument_type_strings = call["argumentTypes"]
        argument0_type_vector = type_to_vector.get(argument_type_strings[0], [0]*type_embedding_size)
        argument1_type_vector = type_to_vector.get(argument_type_strings[1], [0]*type_embedding_size)
        if (self.is_known_type(argument_type_strings[0]) or self.is_known_type(argument_type_strings[1])):
            self.stats["calls_with_known_types"] += 1
        if (self.is_known_type(argument_type_strings[0]) and self.is_known_type(argument_type_strings[1])):
            self.stats["calls_with_both_known_types"] += 1
        
        parameter_strings = call["parameters"]
        # parameter0_vector = name_to_vector.get(parameter_strings[0], [0]*name_embedding_size)
        # parameter1_vector = name_to_vector.get(parameter_strings[1], [0]*name_embedding_size)
        if (parameter_strings[0] in name_to_vector or parameter_strings[1] in name_to_vector):
            self.stats["calls_with_known_parameters"] += 1
        
        x_keep = list(itertools.chain.from_iterable(correct_vectors))
        if call["base"] == "":
            x_keep = [0] * (2 * name_embedding_size) + x_keep
        # for all xy-pairs: y value = probability that incorrect
        # x_keep = callee_vector + argument0_vector + argument1_vector
        # x_keep += base_vector + argument0_type_vector + argument1_type_vector
        # x_keep += parameter0_vector + parameter1_vector #+ file_name_vector
        y_keep = [0]
        # print("callee_vector: ", callee_vector)
        # print("argument0_vector: ", argument0_vector)
        # print("argument1_vector: ", argument1_vector)
        # print("base_vector: ", base_vector)
        # print("argument0_type_vector: ", argument0_type_vector)
        # print("argument1_type_vector: ", argument1_type_vector)
        # print("parameter0_vector: ", parameter0_vector)
        # print("parameter1_vector: ", parameter1_vector)
        # print("\n\n")
        xs.append(x_keep)
        ys.append(y_keep)
        if calls != None:
            calls.append(CodePiece(callee_string, argument_strings, call["src"]))
        
        x_swap = list(itertools.chain.from_iterable(wrong_vectors))
        if call["base"] == "":
            x_swap = [0] * (2 * name_embedding_size) + x_swap
        
        print(correct_code, len(x_keep), buggy_code, len(x_swap))
        # x_swap = callee_vector + argument1_vector + argument0_vector
        # x_swap += base_vector + argument1_type_vector + argument0_type_vector
        # x_swap += parameter0_vector + parameter1_vector #+ file_name_vector
        y_swap = [1]
        xs.append(x_swap)
        ys.append(y_swap)
        if calls != None:
            calls.append(CodePiece(callee_string, argument_strings, call["src"]))
            
    
    def anomaly_score(self, y_prediction_orig, y_prediction_changed):
        return y_prediction_orig - y_prediction_changed # higher means more likely to be anomaly in current code
    
    def normal_score(self, y_prediction_orig, y_prediction_changed):
        return y_prediction_changed - y_prediction_orig # higher means more likely to be correct in current code
