'''
Created on Jun 23, 2017

@author: Rafael Karampatsis
Based on code by Michael Pradel
'''

import sys
import json
import os
from os.path import join
from os import getcwd
from collections import Counter, namedtuple
import math
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
import random
import time
import numpy as np
import Util
import LearningDataSwappedArgs
import LearningDataBinOperator
import LearningDataSwappedBinOperands
import LearningDataIncorrectBinaryOperand
import LearningDataIncorrectAssignment
import LearningDataMissingArg
import queue
import threading
import traceback

from threading import Thread
from Util import *
from bilm import TokenBatcher
from ELMoUtil import *
from ELMoClient import *

name_embedding_size = 200
file_name_embedding_size = 50
type_embedding_size = 5
node_type_embedding_size = 8 # if changing here, then also change in LearningDataBinOperator

Anomaly = namedtuple("Anomaly", ["message", "score"])

# Number of training epochs
EPOCHS = 10
# Number of threads 
BATCHING_THREADS = 1
# Minibatch size. An even number is mandatory. A power of two is advised (for optimization purposes).
BATCH_SIZE = 63 #256
# assert BATCH_SIZE % 2 == 0, "Batch size must be an even number."
# Queue used to store code_pieces from_which minibatches are generated
CODE_PIECES_QUEUE_SIZE = 1000000
code_pieces_queue = queue.Queue(maxsize=CODE_PIECES_QUEUE_SIZE)
# Queue used to store generated minibatches
BATCHES_QUEUE_SIZE = 8192
batches_queue = queue.Queue(maxsize=BATCHES_QUEUE_SIZE)

max_tokens_threshold = 30
USE_ELMO = True
# Connecting to ELMo server
# socket = connect('localhost', PORT)


def parse_data_paths(args):
    training_data_paths = []
    eval_data_paths = []
    mode = None
    for arg in args:
        if arg == "--trainingData":
            assert mode == None
            mode = "trainingData"
        elif arg == "--validationData":
            assert mode == "trainingData"
            mode = "validationData"
        else:
            path = join(getcwd(), arg)
            if mode == "trainingData":
                training_data_paths.append(path)
            elif mode == "validationData":
                eval_data_paths.append(path)
            else:
                print("Incorrect arguments")
                sys.exit(0)
    return [training_data_paths, eval_data_paths]

def prepare_xy_pairs(data_paths, learning_data):
    xs = []
    ys = []
    code_pieces = [] # keep calls in addition to encoding as x,y pairs (to report detected anomalies)
    
    for code_piece in Util.DataReader(data_paths):
        learning_data.code_to_xy_pairs(code_piece, xs, ys, name_to_vector, type_to_vector, node_type_to_vector, code_pieces)
        # print(code_piece)
        # learning_data.code_to_ELMo_xy_pairs(code_piece, xs, ys, name_to_vector, type_to_vector, node_type_to_vector, socket, code_pieces)
    x_length = len(xs[0])

    print("Stats: " + str(learning_data.stats))
    print("Number of x,y pairs: " + str(len(xs)))
    print("Length of x vectors: " + str(x_length))
    xs = np.array(xs)
    ys = np.array(ys)
    return [xs, ys, code_pieces]

def prepare_xy_pairs_batches(data_paths, learning_data):
    for code_piece in Util.DataReader(data_paths, False):
        code_pieces_queue.put((code_piece, learning_data))

def batch_generator(ELMoModel):
    try:
        xs = []
        ys = []
        code_pieces = []
        while True:
            # print('Asked for code_piece')
            queue_entry = code_pieces_queue.get()
            if queue_entry is None:
                if len(xs) > 0:
                    batch = [np.array(xs), np.array(ys)]
                    batches_queue.put(batch)
                code_pieces_queue.task_done()
                break
            code_piece, learning_data = queue_entry
            if len(code_piece["tokens"]) <= max_tokens_threshold:
                code_pieces.append(code_piece)
                if len(code_pieces) == BATCH_SIZE:
                    if USE_ELMO:
                        learning_data.code_to_ELMo_xy_pairs(code_pieces, xs, ys, name_to_vector, \
                            type_to_vector, node_type_to_vector, ELMoModel, None)
                    else:
                        for code in code_pieces:
                            learning_data.code_to_xy_pairs(code, xs, ys, name_to_vector, type_to_vector, node_type_to_vector, None)
                    code_pieces = []
                    if len(xs) > 0:
                        batch = [np.array(xs), np.array(ys)]
                        batches_queue.put(batch)
                    xs = []
                    ys = []
            # print('Got code piece:', code_piece)
            
            # Create minibatches
            # code_pieces = None #[] # keep calls in addition to encoding as x,y pairs (to report detected anomalies)        
            code_pieces_queue.task_done()
    except Exception as e:
        print('Exception:', str(e))
        exc_type, exc_obj, tb = sys.exc_info()
        print(traceback.format_exc())
        code_pieces_queue.task_done()

def sample_xy_pairs(xs, ys, number_buggy):
    sampled_xs = []
    sampled_ys = []
    buggy_indices = []
    for i, y in enumerate(ys):
        if y == [1]:
            buggy_indices.append(i)
    sampled_buggy_indices = set(np.random.choice(buggy_indices, size=number_buggy, replace=False))
    for i, x in enumerate(xs):
        y = ys[i]
        if y == [0] or i in sampled_buggy_indices:
            sampled_xs.append(x)
            sampled_ys.append(y)
    return sampled_xs, sampled_ys

def create_keras_network(dimensions):
    # simple feedforward network
    model = Sequential()
    # model.add(Dropout(0.2, input_shape=(x_length,)))
    model.add(Dropout(0.2, input_shape=(dimensions,)))
    # model.add(Dense(200, input_dim=x_length, activation="relu", kernel_initializer='normal'))
    model.add(Dense(200, input_dim=dimensions, activation="relu", kernel_initializer='normal'))
    model.add(Dropout(0.2))
    #model.add(Dense(200, activation="relu"))
    model.add(Dense(1, activation="sigmoid", kernel_initializer='normal'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model



if __name__ == '__main__':
    # arguments (for learning new model): what --learn <name to vector file> <type to vector file> <AST node type to vector file> --trainingData <list of call data files> --validationData <list of call data files>
    # arguments (for learning new model): what --load <model file> <name to vector file> <type to vector file> <AST node type to vector file> --trainingData <list of call data files> --validationData <list of call data files>
    #   what is one of: SwappedArgs, BinOperator, SwappedBinOperands, IncorrectBinaryOperand, IncorrectAssignment
    print("BugDetection started with " + str(sys.argv))
    time_start = time.time()
    what = sys.argv[1]
    option = sys.argv[2]
    if option == "--learn":
        name_to_vector_file = join(getcwd(), sys.argv[3])
        type_to_vector_file = join(getcwd(), sys.argv[4])
        node_type_to_vector_file = join(getcwd(), sys.argv[5])
        training_data_paths, validation_data_paths = parse_data_paths(sys.argv[6:])
    elif option == "--load":
        print("--load option is buggy and currently disabled")
        sys.exit(1)
        model_file = sys.argv[3]
        name_to_vector_file = join(getcwd(), sys.argv[4])
        type_to_vector_file = join(getcwd(), sys.argv[5])
        node_type_to_vector_file = join(getcwd(), sys.argv[6])
        training_data_paths, validation_data_paths = parse_data_paths(sys.argv[7:])
    else:
        print("Incorrect arguments")
        sys.exit(1)
    
    with open(name_to_vector_file) as f:
        name_to_vector = json.load(f)
    with open(type_to_vector_file) as f:
        type_to_vector = json.load(f)
    with open(node_type_to_vector_file) as f:
        node_type_to_vector = json.load(f)
    
    if what == "SwappedArgs":
        learning_data = LearningDataSwappedArgs.LearningData()
    elif what == "BinOperator":
        learning_data = LearningDataBinOperator.LearningData()
    elif what == "SwappedBinOperands":
        learning_data = LearningDataSwappedBinOperands.LearningData()
    elif what == "IncorrectBinaryOperand":
        learning_data = LearningDataIncorrectBinaryOperand.LearningData()
    elif what == "IncorrectAssignment":
        learning_data = LearningDataIncorrectAssignment.LearningData()
    elif what == "MissingArg":
        learning_data = LearningDataMissingArg.LearningData()
    else:
        print("Incorrect argument for 'what'")
        sys.exit(1)
    
    print("Statistics on training data:")
    learning_data.pre_scan(training_data_paths, validation_data_paths)
    # prepare x,y pairs for learning and validation
    
    # prepare_xy_pairs_batches(training_data_paths, learning_data)
    # xs_training, ys_training, _ = prepare_xy_pairs_batches(training_data_paths, learning_data)
    # x_length = len(xs_training[0])
    # print("Training examples   : " + str(len(xs_training)))
    # print(learning_data.stats)
    
    # manual validation of stored model (for debugging)
    if option == "--load":
        model = load_model(model_file)
        print("Loaded model.")
    elif option == "--learn": 
        # Calculate model dimensions
        model_dimensions = 0
        if what == "SwappedArgs":
            if USE_ELMO:
                dimensions = (max_tokens_threshold + 2) * name_embedding_size #+ 2 * type_embedding_size
                dimensions = 200
            else:
                dimensions = 6 * name_embedding_size + 2 * type_embedding_size
        elif what == "BinOperator":
            if USE_ELMO:
                dimensions = max_tokens_threshold * name_embedding_size + 2 * type_embedding_size + 2 * node_type_embedding_size
            else:
                dimensions = 2 * name_embedding_size + len(learning_data.all_operators) + 2 * type_embedding_size + 2 * node_type_embedding_size
        elif what == "IncorrectBinaryOperand":
            if USE_ELMO:
                dimensions = max_tokens_threshold * name_embedding_size + 2 * type_embedding_size + 2 * node_type_embedding_size
            else:
                dimensions = 2 * name_embedding_size + len(learning_data.all_operators) + 2 * type_embedding_size + 2 * node_type_embedding_size
        
        if dimensions == 0:
            sys.exit(1)
        
        session = tf.keras.backend.get_session()
        vocab_file = 'ELMoVocab.txt'
        # Location of pretrained LM.  Here we use the test fixtures.
        model_dir = '/disk/scratch/mpatsis/eddie/models/phog/js/elmo/emb100_hidden1024_steps20_drop0.1/'
        options_file = os.path.join(model_dir, 'query_options.json')
        weight_file = os.path.join(model_dir, 'weights/weights.hdf5')
        
        # Dump the token embeddings to a file. Run this once for your dataset.
        token_embedding_file = 'elmo_token_embeddings.hdf5'
        # Create a TokenBatcher to map text to token ids.
        batcher = TokenBatcher(vocab_file)

        # Create ELMo Token operation
        elmo_token_op, code_token_ids = create_token_ELMo(options_file, weight_file, \
            token_embedding_file, True)
        ELMoModel = ELMoModel(session, batcher, elmo_token_op, code_token_ids)
        session.run(tf.global_variables_initializer())

        # Create the model
        model = create_keras_network(dimensions)
        
        # Create threads for batch generation
        threads = []
        for i in range(BATCHING_THREADS):
            t = threading.Thread(target=batch_generator, args=(ELMoModel,))
            t.start()
            threads.append(t)
        print("Created %d threads" % BATCHING_THREADS)

        # Train loop for requested number of epochs
        # Retrieve batches from the queue and train on the until they are exausted.
        # A timeout is used to check if the queue is empty.
        time_stamp = math.floor(time.time() * 1000)
        for e in range(1, EPOCHS + 1, 1):
            train_losses = []
            train_accuracies = []
            train_batch_sizes = []
            train_instances = 0
            train_batches = 0
            print("Preparing xy pairs for training data:")
            learning_data.resetStats()
            t = threading.Thread(target=prepare_xy_pairs_batches, args=(training_data_paths, learning_data,))
            t.start()
            # Wait until the batches queue is not empty
            while batches_queue.empty():
                # print('Empty batch queue')
                continue
            try:
                created_model = False
                while True:
                    batch = batches_queue.get(timeout=30)
                    batch_x, batch_y = batch
                    # print("a batch")
                    # if e == 1 and not created_model:
                    #     dimensions = len(batch_x[0])
                    #     model = create_keras_network(dimensions)
                    #     created_model = True

                    batch_len = len(batch_x)
                    print('Batch len:', batch_len)
                    train_instances += batch_len
                    train_batch_sizes.append(batch_len)
                    train_batches += 1
                    print('Batches done:', train_batches)
                    batch_loss, batch_accuracy = model.train_on_batch(batch_x, batch_y)
                    train_losses.append(batch_loss) #* (batch_len / float(BATCH_SIZE))
                    train_accuracies.append(batch_accuracy)
                    print('Batch accuracy:', batch_accuracy)
                    
                    if train_batches % 100 == 0:
                        print("100 batches") 
                        # print(batch_loss, batch_accuracy, mean(train_losses, train_batch_sizes))

                    batches_queue.task_done()
            except queue.Empty:
                pass
            finally:
                # block untill all minibatches have been assigned to a batch_generator thread
                print('Before join in finally')
                code_pieces_queue.join()
                print('After join in finally')

                train_loss = mean(train_losses, train_batch_sizes)
                train_accuracy = mean(train_accuracies, train_batch_sizes)
                print("Epoch %d Training instances %d - Loss & Accuracy [%f, %f]" % \
                    (e, train_instances, train_loss, train_accuracy))
            if e == 1: print(learning_data.stats)
        # stop workers
        for i in range(BATCHING_THREADS):
            code_pieces_queue.put(None)
        for t in threads:
            t.join()

        # for xs in xs_training:
        #     print(len(xs))
        # history = model.fit(xs_training, ys_training, batch_size=100, epochs=10, verbose=1)
        model.save("bug_detection_model_"+str(time_stamp))
    
    time_learning_done = time.time()
    print("Time for learning (seconds): " + str(round(time_learning_done - time_start)))
    
    print("Preparing xy pairs for validation data:")
    learning_data.resetStats()

    # Restart the queues
    code_pieces_queue = queue.Queue(maxsize=CODE_PIECES_QUEUE_SIZE)
    batches_queue = queue.Queue(maxsize=BATCHES_QUEUE_SIZE)

    # Evaluate the model on test data.
    # Create threads for batch generation
    threads = []
    for i in range(BATCHING_THREADS):
        t = threading.Thread(target=batch_generator, args=(ELMoModel,))
        t.start()
        threads.append(t)
    
    test_losses = []
    test_accuracies = []
    test_batch_sizes = []
    test_instances = 0
    test_batches = 0
    t = threading.Thread(target=prepare_xy_pairs_batches, args=(validation_data_paths, learning_data,))
    t.start()
    # prepare_xy_pairs_batches(validation_data_paths, learning_data)
    # Wait until the batches queue is not empty
    while batches_queue.empty():
        continue
    try:
        while True:
            batch = batches_queue.get(timeout=5)
            batch_x, batch_y = batch
            batch_len = len(batch_x)
            test_instances += batch_len
            test_batches += 1
            test_batch_sizes.append(batch_len)
            batch_loss, batch_accuracy = model.test_on_batch(batch_x, batch_y)
            test_losses.append(batch_loss) #* (batch_len / float(BATCH_SIZE))
            test_accuracies.append(batch_accuracy)
            batches_queue.task_done()
    except queue.Empty:
        pass
    finally:
        # block untill all minibatches have been assigned to a batch_generator thread
        code_pieces_queue.join()
        print(learning_data.stats)
        test_loss = mean(test_losses, test_batch_sizes)
        test_accuracy = mean(test_accuracies, test_batch_sizes)
        print("Test instances %d - Loss & Accuracy [%f, %f]" % \
                    (test_instances, test_loss, test_accuracy))
    # stop workers
    for i in range(BATCHING_THREADS):
        code_pieces_queue.put(None)
    for t in threads:
        t.join()

    # Restart the queues
    code_pieces_queue = queue.Queue(maxsize=CODE_PIECES_QUEUE_SIZE)
    batches_queue = queue.Queue(maxsize=BATCHES_QUEUE_SIZE)

    # Evaluate the model on test data.
    # Create threads for batch generation
    threads = []
    for i in range(BATCHING_THREADS):
        t = threading.Thread(target=batch_generator, args=(ELMoModel,))
        t.start()
        threads.append(t)
    
    test_losses = []
    test_accuracies = []
    test_batch_sizes = []
    test_instances = 0
    test_batches = 0
    t = threading.Thread(target=prepare_xy_pairs_batches, args=(training_data_paths, learning_data,))
    t.start()
    # prepare_xy_pairs_batches(validation_data_paths, learning_data)
    # Wait until the batches queue is not empty
    while batches_queue.empty():
        continue
    try:
        while True:
            batch = batches_queue.get(timeout=5)
            batch_x, batch_y = batch
            batch_len = len(batch_x)
            test_instances += batch_len
            test_batches += 1
            test_batch_sizes.append(batch_len)
            batch_loss, batch_accuracy = model.test_on_batch(batch_x, batch_y)
            test_losses.append(batch_loss) #* (batch_len / float(BATCH_SIZE))
            test_accuracies.append(batch_accuracy)
            batches_queue.task_done()
    except queue.Empty:
        pass
    finally:
        # block untill all minibatches have been assigned to a batch_generator thread
        code_pieces_queue.join()
        print(learning_data.stats)
        test_loss = mean(test_losses, test_batch_sizes)
        test_accuracy = mean(test_accuracies, test_batch_sizes)
        print("Train instances %d - Loss & Accuracy [%f, %f]" % \
                    (test_instances, test_loss, test_accuracy))
    # stop workers
    for i in range(BATCHING_THREADS):
        code_pieces_queue.put(None)
    for t in threads:
        t.join()

    # xs_validation, ys_validation, code_pieces_validation = prepare_xy_pairs(validation_data_paths, learning_data)
    # print("Validation examples : " + str(len(xs_validation)))

    # All the representations were received so close the socket.
    # socket.sendall(CONN_END)
    # socket.close()
    
    # print()
    # print("Validation loss & accuracy: " + str(validation_loss))
    sys.exit(0)

    # compute precision and recall with different thresholds for reporting anomalies
    # assumption: correct and swapped arguments are alternating in list of x-y pairs
    threshold_to_correct = Counter()
    threshold_to_incorrect = Counter()
    threshold_to_found_seeded_bugs = Counter()
    threshold_to_warnings_in_orig_code = Counter()
    ys_prediction = model.predict(xs_validation)
    poss_anomalies = []
    for idx in range(0, len(xs_validation), 2):
        y_prediction_orig = ys_prediction[idx][0] # probab(original code should be changed), expect 0
        y_prediction_changed = ys_prediction[idx + 1][0] # probab(changed code should be changed), expect 1
        anomaly_score = learning_data.anomaly_score(y_prediction_orig, y_prediction_changed) # higher means more likely to be anomaly in current code
        normal_score = learning_data.normal_score(y_prediction_orig, y_prediction_changed) # higher means more likely to be correct in current code
        is_anomaly = False
        for threshold_raw in range(1, 20, 1):
            threshold = threshold_raw / 20.0
            suggests_change_of_orig = anomaly_score >= threshold
            suggests_change_of_changed = normal_score >= threshold
            # counts for positive example
            if suggests_change_of_orig:
                threshold_to_incorrect[threshold] += 1
                threshold_to_warnings_in_orig_code[threshold] += 1
            else:
                threshold_to_correct[threshold] += 1
            # counts for negative example
            if suggests_change_of_changed:
                threshold_to_correct[threshold] += 1
                threshold_to_found_seeded_bugs[threshold] += 1
            else:
                threshold_to_incorrect[threshold] += 1
            
            # check if we found an anomaly in the original code
            if suggests_change_of_orig:
                is_anomaly = True
                
        if is_anomaly:
            code_piece = code_pieces_validation[idx]
            message = "Score : " + str(anomaly_score) + " | " + code_piece.to_message()
#             print("Possible anomaly: "+message)
            # Log the possible anomaly for future manual inspection
            poss_anomalies.append(Anomaly(message, anomaly_score))
    
    f_inspect = open('poss_anomalies.txt', 'w+')
    poss_anomalies = sorted(poss_anomalies, key=lambda a: -a.score)
    for anomaly in poss_anomalies:
        f_inspect.write(anomaly.message + "\n")
    print("Possible Anomalies written to file : poss_anomalies.txt")
    f_inspect.close()

    time_prediction_done = time.time()
    print("Time for prediction (seconds): " + str(round(time_prediction_done - time_learning_done)))
    
    print()
    for threshold_raw in range(1, 20, 1):
        threshold = threshold_raw / 20.0
        recall = (threshold_to_found_seeded_bugs[threshold] * 1.0) / (len(xs_validation) / 2)
        precision = 1 - ((threshold_to_warnings_in_orig_code[threshold] * 1.0) / (len(xs_validation) / 2))
        if threshold_to_correct[threshold] + threshold_to_incorrect[threshold] > 0:
            accuracy = threshold_to_correct[threshold] * 1.0 / (threshold_to_correct[threshold] + threshold_to_incorrect[threshold])
        else:
            accuracy = 0.0
        print("Threshold: " + str(threshold) + "   Accuracy: " + str(round(accuracy, 4)) + "   Recall: " + str(round(recall, 4))+ "   Precision: " + str(round(precision, 4))+"  #Warnings: "+str(threshold_to_warnings_in_orig_code[threshold]))
    

    
    