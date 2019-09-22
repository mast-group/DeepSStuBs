'''
Created on Nov 9, 2017

@author: Michael Pradel
'''

import itertools
import numpy as np
import Util
import sys
from collections import Counter

from ELMoClient import *
from ELMoUtil import ELMoMode
from Util import clean_string


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
                      "calls_with_known_parameters": 0}
    
    def pre_scan(self, training_data_paths, validation_data_paths):
        print("Stats on training data")
        self.gather_stats(training_data_paths)
        print("Stats on validation data")
        self.gather_stats(validation_data_paths)

    def gather_stats(self, data_paths):
        callee_to_freq = Counter()
        argument_to_freq = Counter()
        
        for call in Util.DataReader(data_paths, False):
            callee_to_freq[call["callee"]] += 1
            for argument in call["arguments"]:
                argument_to_freq[argument] += 1
            
        print("Unique callees        : " + str(len(callee_to_freq)))
        print("  " + "\n  ".join(str(x) for x in callee_to_freq.most_common(10)))
        Util.analyze_histograms(callee_to_freq)
        print("Unique arguments      : " + str(len(argument_to_freq)))
        print("  " + "\n  ".join(str(x) for x in argument_to_freq.most_common(10)))
        Util.analyze_histograms(argument_to_freq)


    def code_to_xy_FastText_pairs(self, call, xs, ys, name_to_vector, type_to_vector, node_type_to_vector, calls=None):
        arguments = call["arguments"]
        self.stats["calls"] += 1
        if len(arguments) != 2:
            # print('Skipping call:')
            # print(call)
            # print()
            return
        self.stats["calls_with_two_args"] += 1
        
        # mandatory information: callee and argument names        
        callee_string = call["callee"]
        argument_strings = call["arguments"]
        all_names_known = True
        if not (callee_string in name_to_vector):
            # print('calle_string %s' % callee_string)
            callee_string = 'UNK'
            all_names_known = False
            # return
        for e, argument_string in enumerate(argument_strings):
            if not (argument_string in name_to_vector):
                argument_strings[e] = 'UNK'
                all_names_known = False
                # return
        # print('will calculate feats')
        if all_names_known: self.stats["calls_with_known_names"] += 1
        callee_vector = list(name_to_vector[callee_string])
        argument0_vector = list(name_to_vector[argument_strings[0]])
        argument1_vector = list(name_to_vector[argument_strings[1]])

        # optional information: base object, argument types, etc.
        base_string = call["base"]
        if len(base_string) > 0:
           base_vector = list(name_to_vector[base_string])
        else:
            base_vector = [0] * name_embedding_size
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
        if parameter_strings[0] in name_to_vector:
            parameter0_vector = list(name_to_vector[parameter_strings[0]])
        else:
            parameter0_vector = [0] * name_embedding_size
        if parameter_strings[1] in name_to_vector:
            parameter1_vector = list(name_to_vector[parameter_strings[1]])
        else:
            parameter1_vector = [0] * name_embedding_size
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


    def mutate(self, call):
        self.stats["calls"] += 1
        mutated_call = dict()

        mutated_call["base"] = call["base"]
        mutated_call["callee"] = call["callee"]
        mutated_call["calleeLocation"] = call["calleeLocation"]

        mutated_call["arguments"] = []
        mutated_call["argumentLocations"] = []
        mutated_call["argumentTypes"] = []
        mutated_call["parameters"] = []
        
        swap_mapping = self.get_swap_mapping(call)
        if len(call["arguments"]) != 2:
            return None
        self.stats["calls_with_two_args"] += 1
        
        for i in range(len(call["arguments"])):
            mutated_call["parameters"].append(call["parameters"][i])
            
            insert_index = i
            if i in swap_mapping:
                insert_index = swap_mapping[i]
            
            mutated_call["arguments"].append(call["arguments"][insert_index])
            mutated_call["argumentLocations"].append(call["argumentLocations"][insert_index])
            mutated_call["argumentTypes"].append(call["argumentTypes"][insert_index])


        mutated_call["src"] = call["src"]
        mutated_call["filename"] = call["filename"]
        if "swappedTokens" in call:
            mutated_call["tokens"] = call["swappedTokens"].copy()
            mutated_call["swappedTokens"] = call["tokens"].copy()
        else:
            mutated_call["tokens"] = call["tokens"].copy()

        return mutated_call


    def get_swap_mapping(self, call):
        swap_mapping = {}
        if ( len( call["arguments"]) == 2 ):
            swap_mapping[0] = 1
            swap_mapping[1] = 0
        
        return swap_mapping
    

    def code_features(self, call, embeddings_model, emb_model_type, type_to_vector, node_type_to_vector):
        if emb_model_type == 'w2v' or emb_model_type == 'FastText':
            if isinstance(call, list):
                feats = []
                for call_inst in call:
                    x = self.code_features(call_inst, embeddings_model, emb_model_type, type_to_vector, node_type_to_vector)
                    feats.append(x)
                return feats

            arguments = call["arguments"]
            assert(len(arguments) == 2)

            base_string = call["base"]
            if base_string == '':
                base_vector = [0] * embeddings_model.get_embedding_dims()
            else:
                base_vector = embeddings_model.get_embedding(base_string)

            callee_string = call["callee"]
            callee_vector = embeddings_model.get_embedding(callee_string)
            
            argument_strings = call["arguments"]
            argument0_vector = embeddings_model.get_embedding(argument_strings[0])
            argument1_vector = embeddings_model.get_embedding(argument_strings[1])

            x = callee_vector + argument0_vector + argument1_vector + base_vector

            argument_type_strings = call["argumentTypes"]
            argument0_type_vector = type_to_vector.get(argument_type_strings[0], [0]*type_embedding_size)
            argument1_type_vector = type_to_vector.get(argument_type_strings[1], [0]*type_embedding_size)
            if (self.is_known_type(argument_type_strings[0]) or self.is_known_type(argument_type_strings[1])):
                self.stats["calls_with_known_types"] += 1
            if (self.is_known_type(argument_type_strings[0]) and self.is_known_type(argument_type_strings[1])):
                self.stats["calls_with_both_known_types"] += 1
            
            parameter_strings = call["parameters"]
            if parameter_strings[0] == '':
                parameter0_vector = [0] * embeddings_model.get_embedding_dims()
            else:
                parameter0_vector = embeddings_model.get_embedding(parameter_strings[0])
            if parameter_strings[1] == '':
                parameter1_vector = [1] * embeddings_model.get_embedding_dims()
            else:
                parameter1_vector = embeddings_model.get_embedding(parameter_strings[1])
            
            x += argument0_type_vector + argument1_type_vector
            x += parameter0_vector + parameter1_vector #+ file_name_vector

        elif emb_model_type == 'ELMo':
            if isinstance(call, list):
                feats = []
                queries = []
                base_vecs = []
                type_vecs = []
                part_indices = []
                for i, call_inst in enumerate(call):
                    if call_inst["arguments"][0] == "function":
                        call_inst["arguments"][0] = "STD:function"
                    if call_inst["arguments"][1] == "function":
                        call_inst["arguments"][1] = "STD:function"
                    if call_inst["arguments"][0] == "LIT:this":
                        call_inst["arguments"][0] = "STD:this"
                    if call_inst["arguments"][1] == "LIT:this":
                        call_inst["arguments"][1] = "STD:this"
                    if call_inst["arguments"][0] == "{}":
                        call_inst["arguments"][0] = "STD:{"
                    if call_inst["arguments"][1] == "{}":
                        call_inst["arguments"][1] = "STD:{"
                    
                    if call_inst["base"] == "function":
                        call_inst["base"] = "STD:function"
                    if call_inst["base"] == "LIT:this":
                        call_inst["base"] = "STD:this"
                    if call_inst["base"] == "{}":
                        call_inst["base"] = "STD:{"
                    

                    query = call_inst["tokens"]
                    if call_inst["base"] == '':
                        base_vec = [0] * embeddings_model.get_embedding_dims() * 2
                    else: base_vec = []
                    # base_vec, query  = self._to_ELMo_heuristic_query(call_inst, embeddings_model)
                    queries.append(query)
                    base_vecs.append(base_vec)
                    
                    diff_index = 0
                    for token, r_token in zip(call_inst["tokens"], call_inst["swappedTokens"]):
                        if token != r_token:
                            break
                        diff_index += 1
                    # print(call_inst["tokens"], call_inst["arguments"])
                    left_index = call_inst["tokens"].index(call_inst["arguments"][0])
                    right_index = call_inst["tokens"].index(call_inst["arguments"][1])
                    callee_index = call_inst["tokens"].index(call_inst["callee"])
                    base_index = 0
                    if call_inst["base"] != '':
                        base_index = call_inst["tokens"].index(call_inst["base"])
                    print(base_index, callee_index, diff_index, left_index, right_index)
                    part_indices.append( [[i, base_index], [i, base_index], [i, callee_index], \
                        [i, diff_index], [i, left_index], [i, right_index]] )

                    # Type vector
                    argument_type_strings = call_inst["argumentTypes"]
                    argument0_type_vector = type_to_vector.get(argument_type_strings[0], [0] * type_embedding_size)
                    argument1_type_vector = type_to_vector.get(argument_type_strings[1], [0] * type_embedding_size)
                    type_vecs.append( argument0_type_vector + argument1_type_vector )

                # for query in queries:
                #     print(query)
                # sys.exit(0)

                part_indices = np.array(part_indices)
                print(part_indices.shape())
                embeds = embeddings_model.get_sequence_embeddings(queries)
                print(embeds)
                
                return embeds, type_vecs, base_vecs, part_indices
                for i in range(len(embeds)):
                    vec = list(embeds[i].ravel())
                    # if len(vec) != 1600: print(len(vec))
                    if len(base_vecs[i]) > 0 and len(vec) < embeddings_model.get_embedding_dims() * 8:
                        feats.append(base_vecs[i] + vec + type_vecs[i])
                    else:
                        feats.append(vec + type_vecs[i])
                
                return feats
            else:
                base_vec, query  = self._to_ELMo_heuristic_query(call, embeddings_model)
                x = base_vec + list(embeddings_model.get_sequence_embeddings([query]).ravel())
                
                if len(x) != 1600:
                    print(len(x), query)
                argument_type_strings = call["argumentTypes"]
                argument0_type_vector = type_to_vector.get(argument_type_strings[0], [0]*type_embedding_size)
                argument1_type_vector = type_to_vector.get(argument_type_strings[1], [0]*type_embedding_size)
                x += argument0_type_vector + argument1_type_vector
                return x

        elif emb_model_type == 'BPE':
            if isinstance(call, list):
                feats = []
                queries = []
                base_vecs = []
                type_vecs = []
                for call_inst in call:
                    base_vec, query  = self._to_ELMo_heuristic_query(call_inst, embeddings_model)
                    queries.append(query)
                    base_vecs.append(base_vec)
                    # Type vector
                    argument_type_strings = call_inst["argumentTypes"]
                    argument0_type_vector = type_to_vector.get(argument_type_strings[0], [0] * type_embedding_size)
                    argument1_type_vector = type_to_vector.get(argument_type_strings[1], [0] * type_embedding_size)
                    type_vecs.append( argument0_type_vector + argument1_type_vector )

                embeds = embeddings_model.get_sequence_embeddings(queries)
                for i in range(len(embeds)):
                    vec = list(embeds[i].ravel())
                    # if len(vec) != 1600: print(len(vec))
                    if len(base_vecs[i]) > 0 and len(vec) < embeddings_model.get_embedding_dims() * 8:
                        feats.append(base_vecs[i] + vec + type_vecs[i])
                    else:
                        feats.append(vec + type_vecs[i])
                
                return feats
            else:
                base_vec, query  = self._to_ELMo_heuristic_query(call, embeddings_model)
                x = base_vec + list(embeddings_model.get_sequence_embeddings([query]).ravel())
                
                argument_type_strings = call["argumentTypes"]
                argument0_type_vector = type_to_vector.get(argument_type_strings[0], [0]*type_embedding_size)
                argument1_type_vector = type_to_vector.get(argument_type_strings[1], [0]*type_embedding_size)
                x += argument0_type_vector + argument1_type_vector
                return x
        else:
            return None
        

        # if calls != None:
        #     calls.append(CodePiece(callee_string, argument_strings, call["src"]))
        
        return x
    

    def _to_ELMo_heuristic_query(self, call, embeddings_model):
        arguments = call["arguments"]
        assert(len(arguments) == 2)
        callee_string = call["callee"]
        argument_strings = call["arguments"]
        
        query = '%s STD:( %s STD:, %s STD:)' % (clean_string(callee_string), clean_string(argument_strings[0]), \
            clean_string(argument_strings[1]))
        
        base_string = call["base"]
        if base_string == '':
            base_vector = [0] * embeddings_model.get_embedding_dims() * 2
            return base_vector, query.split()
        else:
            query = ('%s STD:. ' % clean_string(base_string)) + query
            return [], query.split()


    def code_to_xy_pairs(self, call, xs, ys, name_to_vector, type_to_vector, node_type_to_vector, calls=None):
        arguments = call["arguments"]
        self.stats["calls"] += 1
        if len(arguments) != 2:
            # print('Skipping call:')
            # print(call)
            # print()
            return
        self.stats["calls_with_two_args"] += 1
        
        # mandatory information: callee and argument names        
        callee_string = call["callee"]
        argument_strings = call["arguments"]
        all_names_known = True
        if not (callee_string in name_to_vector):
            # print('calle_string %s' % callee_string)
            callee_string = 'UNK'
            all_names_known = False
            # return
        for e, argument_string in enumerate(argument_strings):
            if not (argument_string in name_to_vector):
                argument_strings[e] = 'UNK'
                all_names_known = False
                # return
        # print('will calculate feats')
        if all_names_known: self.stats["calls_with_known_names"] += 1
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
        type_representations = []
        for call in func_calls:
            arguments = call["arguments"]
            self.stats["calls"] += 1
            if len(arguments) != 2:
                continue
            self.stats["calls_with_two_args"] += 1
            queries.append(call["tokens"])
            queries.append(call["swappedTokens"])

            argument_type_strings = call["argumentTypes"]
            argument0_type_vector = type_to_vector.get(argument_type_strings[0], [0]*type_embedding_size)
            argument1_type_vector = type_to_vector.get(argument_type_strings[1], [0]*type_embedding_size)
            if (self.is_known_type(argument_type_strings[0]) or self.is_known_type(argument_type_strings[1])):
                self.stats["calls_with_known_types"] += 1
            if (self.is_known_type(argument_type_strings[0]) and self.is_known_type(argument_type_strings[1])):
                self.stats["calls_with_both_known_types"] += 1
            type_representations.append(argument0_type_vector + argument1_type_vector)
            type_representations.append(argument1_type_vector + argument0_type_vector)
        type_representations.append(None)

        if len(queries) == 0:
            return
        queries.append([""] * 30)
        
        elmo_representations = ELMoModel.query(queries, ELMoMode.ALL)
        for i, features in enumerate(zip(elmo_representations, type_representations)):
            representation, type_feats = features
            if i == len(elmo_representations) - 1: break
            xs.append(np.append(representation, np.array(type_feats)))
            ys.append([i % 2])
        
        # if calls != None:
        #     calls.append(CodePiece(callee_string, argument_strings, call["src"]))

    def code_to_ELMo_baseline_xy_pairs(self, func_calls, xs, ys, name_to_vector, type_to_vector, node_type_to_vector, ELMoModel, calls=None):
        queries = []
        type_representations = []
        for call in func_calls:
            arguments = call["arguments"]
            self.stats["calls"] += 1
            if len(arguments) != 2:
                continue
            self.stats["calls_with_two_args"] += 1
            
            # mandatory information: callee and argument names        
            callee_string = call["callee"]
            argument_strings = call["arguments"]
            if not (callee_string in name_to_vector):
                self.stats["calls_with_known_names"] += 1
            
            correct_code = ""
            if call["base"] != "":
                correct_code += call["base"].replace(' ', 'U+0020') + " . "
            buggy_code = correct_code
            correct_code += "%s ( %s , %s )" % (clean_string(callee_string), \
                clean_string(argument_strings[0]), clean_string(argument_strings[1]))
            buggy_code += "%s ( %s , %s )" % (clean_string(callee_string), \
                clean_string(argument_strings[1]), clean_string(argument_strings[0]))
            # correct_query = []
            # buggy_query = []
            if call["base"] != "":
                correct_query = correct_code.split(' ')
                buggy_query = buggy_code.split(' ')
            else:
                correct_query = ["", "."] + correct_code.split(' ')
                buggy_query = ["", "."] + buggy_code.split(' ')
            if len(correct_query) > 8:
                print(correct_query, call)
            queries.append(correct_query)
            queries.append(buggy_query)
        
        if len(queries) == 0:
            return
        elmo_representations = ELMoModel.query(queries, ELMoMode.ALL)

        for i, features in enumerate(elmo_representations):
            # representation, type_feats = features
            xs.append(features)
            ys.append([i % 2])
    
    
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
