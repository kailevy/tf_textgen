import argparse
import logging
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

from util import Progbar, get_minibatches, minibatches
from model import Model

class TextGen(Model):
    def add_placeholders(self):
        """
        """
        self.input_placeholder = tf.placeholder(tf.int32,
                                                shape=(None, self.config.vocab_size))
        self.labels_placeholder = tf.placeholder(tf.float32,
                                                 shape=(None, self.config.vocab_size))
        self.dropout_placeholder = tf.placeholder(tf.float32, [])

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=0):
        """
        """

        feed_dict = {
            self.input_placeholder: inputs_batch,
            self.labels_placeholder: labels_batch,
            self.dropout_placeholder: dropout
        }
        return feed_dict

    def add_prediction_op(self):
        """
        """
        
        cells = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.config.hidden_dim) for _ in range(self.config.layer_num)])
        x = self.input_placeholder
        #why state/ output
        print ('cells', cells)
        print ('x', x)
        x = tf.cast(x,tf.float32)
        x = tf.expand_dims(x, axis = 2)
        output, state = tf.nn.dynamic_rnn(cells, x, dtype=tf.float32)
        preds = tf.sigmoid(output)
        return preds

    def add_loss_op(self, preds):
        """
        """

        y = self.labels_placeholder
        loss = tf.losses.softmax_cross_entropy(preds - y)
        loss = tf.reduce_mean(loss)
        return loss

    def add_training_op(self, loss):
        """
        """

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.lr)

        grads_and_vars = optimizer.compute_gradients(loss)
        variables = [output[1] for output in grads_and_vars]
        gradients = [output[0] for output in grads_and_vars]
        if self.config.clip_gradients:
            tmp_gradients = tf.clip_by_global_norm(gradients, clip_norm=self.config.max_grad_norm)[0]
            gradients = tmp_gradients

        grads_and_vars = [(gradients[i], variables[i]) for i in range(len(gradients))]
        self.grad_norm = tf.global_norm(gradients)

        train_op = optimizer.apply_gradients(grads_and_vars)

        assert self.grad_norm is not None, "grad_norm was not set properly!"
        return train_op
        

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """
        """

        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss, grad_norm = sess.run([self.train_op, self.loss, self.grad_norm], feed_dict=feed)
        return loss, grad_norm

    def run_epoch(self, sess, train, label):
        """
        """

        prog = Progbar(target=1 + int(len(train) / self.config.batch_size))
        losses, grad_norms = [], []
        for inputs_minibatch, labels_minibatch in get_minibatches([train, label], self.config.batch_size):
        #for i, batch in enumerate(minibatches(train, label, self.config.batch_size)):
            loss, grad_norm = self.train_on_batch(sess, inputs_minibatch, labels_minibatch)
            losses.append(loss)
            grad_norms.append(grad_norm)
            prog.update(i + 1, [("train loss", loss)])

    def fit(self, sess, train):
        """
        """

        losses, grad_norms = [], []
        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            loss, grad_norm = self.run_epoch(sess, train)
            losses.append(loss)
            grad_norms.append(grad_norm)

        return losses, grad_norms

    def __init__(self, config):
        self.config = config
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.build()
