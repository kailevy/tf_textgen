import argparse
import logging
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

from util import Progbar, minibatches
from model import Model

class Config:
    """
    """
    batch_size = 100
    n_epochs = 40
    lr = 0.2

class TextGen(Model):
    def add_placeholders(self):
        """
        """
        pass

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """
        """
        pass

    def add_prediction_op(self):
        """
        """
        pass

    def add_loss_op(self, preds):
        """
        """
        pass

    def add_training_op(self, loss):
        """
        """
        pass

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """
        """
        pass

    def run_epoch(self, sess, train):
        """
        """
        pass

    def fit(self, sess, train):
        """
        """
        pass

    def __init__(self, config):
        self.config = config
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.build()
