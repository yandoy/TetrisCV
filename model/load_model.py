import numpy as np
import tensorflow as tf
from alexnet import alexnet

WIDTH = 60
HEIGHT = 60
LR = 1e-3
EPOCHS = 5
MODEL_NAME = 'pyTetris2.0-epochs.model'.format(LR, 'alexnet', EPOCHS)
tf.reset_default_graph()
model = alexnet(WIDTH, HEIGHT, LR)


def load_model():
    global model
    model.load(MODEL_NAME, weights_only=True)

    global graph
    graph = tf.get_default_graph()


def return_graph():
    return graph

