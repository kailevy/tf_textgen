from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import argparse
from RNN_utils import *
import tensorflow as tf
from textgen_model import *

class Config:
    """                                                                                               
    """
    batch_size = 100
    n_epochs = 40
    lr = 0.2
    vocab_size = None
    hidden_dim = 500
    layer_num = 2
    clip_gradients = False

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

# Parsing arguments for Network definition
ap = argparse.ArgumentParser()
ap.add_argument('-data_dir', default='./data/test.txt')
ap.add_argument('-batch_size', type=int, default=50)
ap.add_argument('-layer_num', type=int, default=2)
ap.add_argument('-seq_length', type=int, default=50)
ap.add_argument('-hidden_dim', type=int, default=500)
ap.add_argument('-generate_length', type=int, default=500)
ap.add_argument('-nb_epoch', type=int, default=20)
ap.add_argument('-mode', default='train')
ap.add_argument('-weights', default='')
args = vars(ap.parse_args())

DATA_DIR = "../text-generator/data/shakespeare_input.txt"
BATCH_SIZE = args['batch_size']
HIDDEN_DIM = args['hidden_dim']
SEQ_LENGTH = args['seq_length']
WEIGHTS = args['weights']

GENERATE_LENGTH = args['generate_length']
LAYER_NUM = args['layer_num']

# Creating training data
X, y, VOCAB_SIZE, ix_to_char, char_to_ix = load_data(DATA_DIR, SEQ_LENGTH)

# Set up some parameters.
config = Config(VOCAB_SIZE)

handler = logging.FileHandler('./output')
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
logging.getLogger().addHandler(handler)

with tf.Graph().as_default():
    print ("Building model...",)
    start = time.time()
    model = TextGen(config)
    print ("took %.2f seconds", time.time() - start)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init)
        model.fit(session, X)

        # Save predictions in a text file.
        output = model.output(session, dev_raw)
        sentences, labels, predictions = zip(*output)
        output = zip(sentences, labels, predictions)

        with open(model.config.conll_output, 'w') as f:
            write_conll(f, output)
        with open(model.config.eval_output, 'w') as f:
            for sentence, labels, predictions in output:
                print_sentence(f, sentence, labels, predictions)


