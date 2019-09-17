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
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.backend.tensorflow_backend import set_session

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

import model.ModelFactory
from model.ModelFactory import *

from gensim.models import FastText
from threading import Thread
from Util import *
from bilm import TokenBatcher
from ELMoUtil import *
from ELMoClient import *

# Set random seed
random.seed(1234)


name_embedding_size = 200
file_name_embedding_size = 50
type_embedding_size = 5
node_type_embedding_size = 8 # if changing here, then also change in LearningDataBinOperator

Anomaly = namedtuple("Anomaly", ["message", "score"])

# Number of training epochs
EPOCHS = 1
# Number of threads 
BATCHING_THREADS = 1
# Minibatch size. An even number is mandatory. A power of two is advised (for optimization purposes).
BATCH_SIZE = 100#63 #256
# assert BATCH_SIZE % 2 == 0, "Batch size must be an even number."
# Queue used to store code_pieces from_which minibatches are generated
CODE_PIECES_QUEUE_SIZE = 1000000
code_pieces_queue = queue.Queue(maxsize=CODE_PIECES_QUEUE_SIZE)

CODE_PAIRS_QUEUE_SIZE = 250000
code_pairs_queue = queue.Queue(maxsize=CODE_PAIRS_QUEUE_SIZE)

# Queue used to store generated minibatches
BATCHES_QUEUE_SIZE = 8192
batches_queue = queue.Queue(maxsize=BATCHES_QUEUE_SIZE)

max_tokens_threshold = 30
USE_ELMO = False
USE_ELMO_TOP_ONLY = True
GRAPH = None
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
    
    for code_piece in Util.DataReader(data_paths, False):
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



def fill_code_pairs_queue(code_pairs):
    for code_pair in code_pairs:
        code_pairs_queue.put(code_pair)
    code_pairs_queue.put(None)


def create_code_pairs(data_paths, learning_data):
    code_pairs = []
    for code_piece in Util.DataReader(data_paths, False):
        buggy_code_piece = learning_data.mutate(code_piece)
        if not buggy_code_piece is None:
            code_pairs.append( (code_piece, buggy_code_piece) )
    return code_pairs


def minibatch_generator():
    xs = []
    ys = []
    code_pieces = []
    while True:
        try:
            code_pair = code_pairs_queue.get()
            if code_pair is None:
                print('Consumed all code pairs.')
                if len(code_pieces) > 0:
                    # Query the model for features
                    xs = learning_data.code_features(code_pieces, embeddings_model, emb_model_type, type_to_vector, node_type_to_vector)
                    # xs = xs.reshape(-1, 600)
                    # xs = tf.reshape(xs, [BATCH_SIZE, 600])
                    # for code_piece in code_pieces:
                    #     x = learning_data.code_features(code_piece, embeddings_model, emb_model_type, type_to_vector, node_type_to_vector)
                    #     xs.append(x)
                    batch = [np.array(xs), np.array(ys)]
                    # batch = [np.array(xs), np.array(ys)]
                    batches_queue.put(batch)
                # code_pairs_queue.task_done()
                break
            fixed, buggy = code_pair
            code_pieces.append(fixed)
            ys.append([0])
            code_pieces.append(buggy)
            ys.append([1])
            if len(code_pieces) == BATCH_SIZE:
                # Query the model for features
                xs = learning_data.code_features(code_pieces, embeddings_model, emb_model_type, type_to_vector, node_type_to_vector)
                # print(xs)
                # xs = xs.reshape(-1, 600)
                # xs = tf.reshape(xs, [BATCH_SIZE, 600])
                # for code_piece in code_pieces:
                #     x = learning_data.code_features(code_piece, embeddings_model, emb_model_type, type_to_vector, node_type_to_vector)
                #     xs.append(x)
                batch = [np.array(xs), np.array(ys)]
                # batch = [np.array(xs), np.array(ys)]
                batches_queue.put(batch)
                xs = []
                ys = []
                code_pieces = []
        except Exception as e:
            print('Exception:', str(e))
            exc_type, exc_obj, tb = sys.exc_info()
            print(traceback.format_exc())
            break
        finally:
            code_pairs_queue.task_done()


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
            if True:
            # if len(code_piece["tokens"]) <= max_tokens_threshold:
                code_pieces.append(code_piece)
                if len(code_pieces) == BATCH_SIZE:
                    if USE_ELMO:
                        learning_data.code_to_ELMo_baseline_xy_pairs(code_pieces, xs, ys, name_to_vector, \
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
    dense_dims = 200
    # dense_dims = 1200
    # simple feedforward network
    model = Sequential()
    # model.add(Dropout(0.2, input_shape=(x_length,)))
    model.add(Dropout(0.2, input_shape=(dimensions,)))
    # model.add(Dense(200, input_dim=x_length, activation="relu", kernel_initializer='normal'))
    model.add(Dense(dense_dims, input_dim=dimensions, activation="relu", kernel_initializer='normal'))
    model.add(Dropout(0.2))
    #model.add(Dense(200, activation="relu"))
    model.add(Dense(1, activation="sigmoid", kernel_initializer='normal'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def create_tf_network(dimensions, inp, extra_dims):
    dense_dims = 200
    # inp = tf.placeholder(shape=[None, dimensions], dtype=tf.float32)
    extra_feats = tf.placeholder(shape=[None, extra_dims], dtype=tf.float32)
    labels = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    drop_inp = tf.nn.dropout(tf.concat([inp, extra_feats], 1), 1 - 0.2)
    nn = tf.layers.dense(drop_inp, dense_dims, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.normal)
    drop_nn = tf.nn.dropout(nn, 1 - 0.2)
    out = tf.layers.dense(drop_nn, 1, activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.normal)
    loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(
        target=labels,
        output=out
    ))
    acc = tf.metrics.accuracy(
        labels= tf.round(labels),
        predictions= tf.round(out)
    )
    optimizer = tf.train.RMSPropOptimizer(0.001, epsilon=1e-07).minimize(loss)
    init = tf.global_variables_initializer()
    tf.summary.scalar("loss", loss)
    merged_summary_op = tf.summary.merge_all()
    return extra_feats, labels, loss, acc, out, optimizer



if __name__ == '__main__':
    # arguments (for learning new model): what --learn <name to vector file> <type to vector file> <AST node type to vector file> --trainingData <list of call data files> --validationData <list of call data files>
    # arguments (for learning new model): what --load <model file> <name to vector file> <type to vector file> <AST node type to vector file> --trainingData <list of call data files> --validationData <list of call data files>
    #   what is one of: SwappedArgs, BinOperator, SwappedBinOperands, IncorrectBinaryOperand, IncorrectAssignment
    # print("BugDetection started with " + str(sys.argv))
    time_start = time.time()
    what = sys.argv[1]
    option = sys.argv[2]
    if option == "--learn":
        # name_to_vector_file = join(getcwd(), sys.argv[3])
        config_file = sys.argv[3]
        type_to_vector_file = join(getcwd(), sys.argv[4])
        node_type_to_vector_file = join(getcwd(), sys.argv[5])
        metrics_file = join(getcwd(), sys.argv[6])
        training_data_paths, validation_data_paths = parse_data_paths(sys.argv[7:])
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
    

    session = tf.Session(config=config)
    set_session(session)  # set this TensorFlow session as the default session for Keras
    # session = tf.keras.backend.get_session()
    
    model_factory = ModelFactory(config_file, session)
    embeddings_model = model_factory.get_model()
    emb_model_type = model_factory.get_model_type()
    name_to_vector = embeddings_model.get_name_to_vector()

    GRAPH = session.graph
    # for op in GRAPH.get_operations():
    #     print(str(op.name))
    

    # with open(name_to_vector_file) as f:
    #     name_to_vector = json.load(f)
    # # m = FastText.load(name_to_vector_file)
    # name_to_vector = m.wv
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
                # dimensions = max_tokens_threshold * name_embedding_size + 2 * type_embedding_size
                # dimensions = name_embedding_size
                # dimensions = name_embedding_size + 2 * type_embedding_size
                dimensions = 8 * name_embedding_size + 2 * type_embedding_size
            else:
                dimensions = 6 * name_embedding_size + 2 * type_embedding_size
        elif what == "BinOperator":
            if USE_ELMO:
                # dimensions = max_tokens_threshold * name_embedding_size #+ 2 * type_embedding_size + 2 * node_type_embedding_size
                # dimensions = 3 * name_embedding_size + len(learning_data.all_operators) + 2 * type_embedding_size + 2 * node_type_embedding_size
                dimensions = 3 * name_embedding_size #+ len(learning_data.all_operators) + 2 * type_embedding_size + 2 * node_type_embedding_size
                extra_dims = len(learning_data.all_operators) + 2 * type_embedding_size + 2 * node_type_embedding_size
            else:
                dimensions = 2 * name_embedding_size + len(learning_data.all_operators) + 2 * type_embedding_size + 2 * node_type_embedding_size
        elif what == "IncorrectBinaryOperand":
            if USE_ELMO:
                dimensions = 3 * name_embedding_size 
                extra_dims = len(learning_data.all_operators) + 2 * type_embedding_size + 2 * node_type_embedding_size
                # dimensions = 3 * name_embedding_size + len(learning_data.all_operators) + 2 * type_embedding_size + 2 * node_type_embedding_size
                # dimensions = max_tokens_threshold * name_embedding_size + 2 * type_embedding_size + 2 * node_type_embedding_size
            else:
                dimensions = 2 * name_embedding_size + len(learning_data.all_operators) + 2 * type_embedding_size + 2 * node_type_embedding_size
        
        if dimensions == 0:
            sys.exit(1)
        

        vocab_file = 'ELMoVocab.txt'
        # Location of pretrained LM.  Here we use the test fixtures.
        model_dir = '/disk/scratch/mpatsis/eddie/models/phog/js/elmo/emb100_hidden1024_steps20_drop0.1/'
        options_file = os.path.join(model_dir, 'query_options.json')
        weight_file = os.path.join(model_dir, 'weights/weights.hdf5')
        
        # # Dump the token embeddings to a file. Run this once for your dataset.
        # token_embedding_file = 'elmo_token_embeddings.hdf5'
        # # Create a TokenBatcher to map text to token ids.
        # batcher = TokenBatcher(vocab_file)

        # Create ELMo Token operation
        # elmo_token_op, code_token_ids = create_token_ELMo(options_file, weight_file, \
        #     token_embedding_file, USE_ELMO_TOP_ONLY)
        # ELMoModel = ELMoModel(session, batcher, elmo_token_op, code_token_ids)
        # session.run(tf.global_variables_initializer())

        # Create the model
        with session.as_default():
            with GRAPH.as_default():
                model = create_keras_network(dimensions)
                # ch_ids = embeddings_model.get_code_character_ids()
                # inp_op = embeddings_model.get_code_rep_op()['weighted_op']
                # print('inp_op=', inp_op)
                # r_inp_op = tf.reshape(inp_op, [-1, dimensions])
                # print('rinp_op=', r_inp_op)
                # extra_feats, labels, loss, acc, out, optimizer = create_tf_network(dimensions, r_inp_op, extra_dims)
                print('Created the model!')
                session.run(tf.global_variables_initializer())
                session.run(tf.local_variables_initializer())
                
                # for op in GRAPH.get_operations():
                #     print(str(op.name))
                learning_data.resetStats()
                train_code_pairs = create_code_pairs(training_data_paths, learning_data)
                test_code_pairs = create_code_pairs(validation_data_paths, learning_data)

                # Training loop
                time_stamp = math.floor(time.time() * 1000)
                for e in range(1, EPOCHS + 1, 1):
                    print("Serving code pairs in the queue.")
                    # Create thread for code pair creation.
                    code_pairs_thread = threading.Thread(target=fill_code_pairs_queue, args=((train_code_pairs), ))
                    code_pairs_thread.start()

                    # Create thread for minibatches creation.
                    batching_thread = threading.Thread(target=minibatch_generator)
                    batching_thread.start()

                    pass
                    # Variables for training stats
                    train_losses = []
                    train_accuracies = []
                    train_batch_sizes = []
                    train_instances = 0
                    train_batches = 0

                    # Wait until the batches queue is not empty
                    while batches_queue.empty():
                        # print('Empty batch queue')
                        continue

                    
                    while True:
                        epoch_batches_done = False
                        try:
                            batch = batches_queue.get(timeout=30)
                            batch_x, batch_y = batch
                            # code_ids, extra_fs = batch_x
                            # print('batch_x:', batch_x.shape)

                            batch_len = len(batch_y)
                            # print('Batch len:', batch_len)
                            train_instances += batch_len
                            train_batch_sizes.append(batch_len)
                            train_batches += 1
                            # print('Batches done:', train_batches)

                            # Train and get loss for minibatch
                            batch_loss, batch_accuracy = model.train_on_batch(batch_x, batch_y)
                            # batch_loss, batch_accuracy, preds, _ = session.run([loss, acc, out, optimizer], \
                            #     feed_dict={ch_ids: code_ids, extra_feats:extra_fs, labels: batch_y})
                            # batch_accuracy = batch_accuracy[0]
                            
                            # print("batch_loss", batch_loss)
                            # print("batch_accuracy", batch_accuracy)
                            # print("preds", preds)
                            train_losses.append(batch_loss) #* (batch_len / float(BATCH_SIZE))
                            train_accuracies.append(batch_accuracy)
                            # print('Batch accuracy:', batch_accuracy)
                            
                            if train_batches % 1000 == 0:
                                print("1000 batches") 
                                print('acc:', mean(train_accuracies, train_batch_sizes))
                                # print(batch_loss, batch_accuracy, mean(train_losses, train_batch_sizes))
                            batches_queue.task_done()
                        except queue.Empty:
                            print('Empty queue')
                            epoch_batches_done = True
                        except Exception as exc:
                            batches_queue.task_done()
                            print(exc)
                            break
                        finally:
                            # block untill all minibatches have been assigned to a batch_generator thread
                            if epoch_batches_done:
                                break
                    print('Before join in finally')
                    code_pairs_thread.join()
                    print('After join in finally')
                    print('Before join in finally')
                    print(code_pairs_queue.empty())
                    print(batches_queue.empty())
                    batching_thread.join()
                    print('After join in finally')

                    train_loss = mean(train_losses, train_batch_sizes)
                    train_accuracy = mean(train_accuracies, train_batch_sizes)
                    print("Epoch %d Training instances %d - Loss & Accuracy [%f, %f]" % \
                        (e, train_instances, train_loss, train_accuracy))
                    if e == 1: print(learning_data.stats)

                    # Wait until both queues have been exhausted.
                    

                time_learning_done = time.time()
                print("Time for learning (seconds): " + str(round(time_learning_done - time_start)))
                
                # Test learned model on test set.
                print("Serving code pairs in the queue.")
                # Create thread for code pair creation.
                code_pairs_thread = threading.Thread(target=fill_code_pairs_queue, args=((test_code_pairs), ))
                code_pairs_thread.start()

                # Create thread for minibatches creation.
                batching_thread = threading.Thread(target=minibatch_generator)
                batching_thread.start()

                predictions = []
                code_pieces_validation = []
                test_losses = []
                test_accuracies = []
                test_batch_sizes = []
                test_instances = 0
                test_batches = 0
                # prepare_xy_pairs_batches(validation_data_paths, learning_data)
                # Wait until the batches queue is not empty
                while batches_queue.empty():
                    continue
                test_batches_done = False
                while True:
                    try:
                        batch = batches_queue.get(timeout=30)
                        batch_x, batch_y = batch
                        # code_ids, extra_fs = batch_x
                        batch_len = len(batch_y)
                        test_instances += batch_len
                        test_batches += 1
                        test_batch_sizes.append(batch_len)
                        batch_loss, batch_accuracy = model.test_on_batch(batch_x, batch_y)
                        # batch_loss, batch_accuracy, preds, _ = session.run([loss, acc, out, optimizer], \
                        #         feed_dict={ch_ids: code_ids, extra_feats:extra_fs, labels: batch_y})
                        # batch_accuracy = batch_accuracy[0]
                        
                        # batch_predictions = model.predict(batch_x)
                        # predictions.extend([pred for pred in batch_predictions])
                        # predictions.extend(model.predict(batch_x))
                        predictions.extend(preds)
                        test_losses.append(batch_loss) #* (batch_len / float(BATCH_SIZE))
                        test_accuracies.append(batch_accuracy)
                        batches_queue.task_done()
                    except queue.Empty:
                        test_batches_done = True
                    finally:
                        # block untill all minibatches have been assigned to a batch_generator thread
                        if test_batches_done:
                            break
                print(learning_data.stats)
                test_loss = mean(test_losses, test_batch_sizes)
                test_accuracy = mean(test_accuracies, test_batch_sizes)
                print("Test instances %d - Loss & Accuracy [%f, %f]" % \
                            (test_instances, test_loss, test_accuracy))
                # stop workers
                code_pairs_thread.join()
                batching_thread.join()

                
                time_prediction_done = time.time()
                print("Time for prediction (seconds): " + str(round(time_prediction_done - time_learning_done)))
                

                #  Test on training batches
                # Create thread for code pair creation.
                code_pairs_thread = threading.Thread(target=fill_code_pairs_queue, args=((train_code_pairs), ))
                code_pairs_thread.start()

                # Create thread for minibatches creation.
                batching_thread = threading.Thread(target=minibatch_generator)
                batching_thread.start()

                train_losses = []
                train_accuracies = []
                train_batch_sizes = []
                train_instances = 0
                train_batches = 0
                # prepare_xy_pairs_batches(validation_data_paths, learning_data)
                # Wait until the batches queue is not empty
                while batches_queue.empty():
                    continue
                train_batches_done = False
                while True:
                    try:
                        batch = batches_queue.get(timeout=30)
                        batch_x, batch_y = batch
                        code_ids, extra_fs = batch_x
                        batch_len = len(batch_y)
                        train_instances += batch_len
                        train_batches += 1
                        train_batch_sizes.append(batch_len)
                        # batch_loss, batch_accuracy = model.train_on_batch(batch_x, batch_y)
                        batch_loss, batch_accuracy, preds, _ = session.run([loss, acc, out, optimizer], \
                                feed_dict={ch_ids: code_ids, extra_feats:extra_fs, labels: batch_y})
                        batch_accuracy = batch_accuracy[0]
                        
                        # batch_predictions = model.predict(batch_x)
                        # predictions.extend([pred for pred in batch_predictions])
                        # predictions.extend(model.predict(batch_x))
                        predictions.extend(preds)
                        train_losses.append(batch_loss) #* (batch_len / float(BATCH_SIZE))
                        train_accuracies.append(batch_accuracy)
                        batches_queue.task_done()
                    except queue.Empty:
                        train_batches_done = True
                    finally:
                        # block untill all minibatches have been assigned to a batch_generator thread
                        if train_batches_done:
                            break
                print(learning_data.stats)
                train_loss = mean(train_losses, train_batch_sizes)
                train_accuracy = mean(train_accuracies, train_batch_sizes)
                print("Train instances %d - Loss & Accuracy [%f, %f]" % \
                            (train_instances, train_loss, train_accuracy))
                # stop workers
                code_pairs_thread.join()
                batching_thread.join()

            # xs_validation, ys_validation, code_pieces_validation = prepare_xy_pairs(validation_data_paths, learning_data)
            # print("Validation examples : " + str(len(xs_validation)))

            # All the representations were received so close the socket.
            # socket.sendall(CONN_END)
            # socket.close()
            
            # print()
            # print("Validation loss & accuracy: " + str(validation_loss))
            
            # sys.exit(0)

            # compute precision and recall with different thresholds for reporting anomalies
            # assumption: correct and swapped arguments are alternating in list of x-y pairs
            threshold_to_correct = Counter()
            threshold_to_incorrect = Counter()
            threshold_to_found_seeded_bugs = Counter()
            threshold_to_warnings_in_orig_code = Counter()
            # ys_prediction = model.predict(xs_validation)
            poss_anomalies = []
            # for idx in range(0, len(xs_validation), 2):
            for idx in range(0, len(predictions), 2):
                y_prediction_orig = predictions[idx][0] # probab(original code should be changed), expect 0
                y_prediction_changed = predictions[idx + 1][0] # probab(changed code should be changed), expect 1
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
                    
    #         if is_anomaly:
    #             code_piece = code_pieces_validation[idx]
    #             message = "Score : " + str(anomaly_score) + " | " + code_piece.to_message()
    # #             print("Possible anomaly: "+message)
    #             # Log the possible anomaly for future manual inspection
    #             poss_anomalies.append(Anomaly(message, anomaly_score))
        
    #     f_inspect = open('poss_anomalies.txt', 'w+')
    #     poss_anomalies = sorted(poss_anomalies, key=lambda a: -a.score)
    #     for anomaly in poss_anomalies:
    #         f_inspect.write(anomaly.message + "\n")
    #     print("Possible Anomalies written to file : poss_anomalies.txt")
    #     f_inspect.close()
        
        print()
        with open(metrics_file, 'w') as f:
            f.write("Train instances %d - Loss & Accuracy [%f, %f]" % \
                        (train_instances, train_loss, train_accuracy) + '\n')
            f.write("Test instances %d - Loss & Accuracy [%f, %f]" % \
                        (test_instances, test_loss, test_accuracy) + '\n')
            f.write('\n')
            
            for threshold_raw in range(1, 20, 1):
                threshold = threshold_raw / 20.0
                recall = (threshold_to_found_seeded_bugs[threshold] * 1.0) / (len(predictions) / 2)
                precision = 1 - ((threshold_to_warnings_in_orig_code[threshold] * 1.0) / (len(predictions) / 2))
                if threshold_to_correct[threshold] + threshold_to_incorrect[threshold] > 0:
                    accuracy = threshold_to_correct[threshold] * 1.0 / (threshold_to_correct[threshold] + threshold_to_incorrect[threshold])
                else:
                    accuracy = 0.0
                print("Threshold: " + str(threshold) + "   Accuracy: " + str(round(accuracy, 4)) + \
                    "   Recall: " + str(round(recall, 4))+ "   Precision: " + str(round(precision, 4)) \
                        + "  #Warnings: " + str(threshold_to_warnings_in_orig_code[threshold]))
                
                f.write(("Threshold: " + str(threshold) + "   Accuracy: " + str(round(accuracy, 4)) + \
                    "   Recall: " + str(round(recall, 4))+ "   Precision: " + str(round(precision, 4)) \
                        + "  #Warnings: " + str(threshold_to_warnings_in_orig_code[threshold])) + '\n')

    
    