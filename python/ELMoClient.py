'''
ELMo client utilities that make queries for code embeddings.

'''

import json
import pickle
import socket
import sys

# Maximum packet size in characters
MAX_PACKET_SIZE = 1000000
# Port to listen in
PORT = 8888
# Finished packet
END = '<END>'.encode()
CONN_END = '<CONN_END>'.encode()


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def connect(server, port):
    """Returns a socket connected to the specified server and port number.
    
    Arguments:
        server {[type]} -- [description]
        port {[type]} -- [description]
    """
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = (server, port)
    eprint('Connecting to %s port %s' % server_address)
    sock.connect(server_address)
    return sock


def query(code, socket, options={'top_layer_only' : False, 'token_embeddings_only' : False}):
    """[summary]
    
    Arguments:
        code {[type]} -- [description]
        socket {[type]} -- [description]
    
    Keyword Arguments:
        options {dict} -- [description] (default: {{'top_layer_only' : False, 'token_embeddings_only' : False}})
    
    Raises:
        ValueError -- [description]
        ValueError -- [description]
    
    Returns:
        [type] -- [description]
    """
    try:
        basestring
    except NameError:
        basestring = str
    if isinstance(code, list):
        for code_sequence in code:
            if not isinstance(code_sequence, basestring): raise ValueError
        code_sequences = [code_sequence.split() for code_sequence in code]
    elif isinstance(code, basestring):
        code_sequences = [code.split()]
    else: raise ValueError
    
    eprint('Sending code sequence "%s"' % code_sequences)
    query = json.dumps({'sequences': code_sequences, 'options': options})
    data = pickle.dumps(query)
    # eprint("Pickled code sequences: ", data)
    socket.sendall(data)
    socket.sendall(END)
    
    received_data = ''.encode()
    received = socket.recv(MAX_PACKET_SIZE)
    while not received[-len(END): ] == END:
        received_data += received
        received = socket.recv(MAX_PACKET_SIZE)
    received_data += received[: -len(END)]
    # eprint('received pickled ELMo embeddings: "%s"' % received_data)
    elmo_representations = pickle.loads(received_data)
    return elmo_representations

