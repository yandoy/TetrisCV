import numpy as np
from alexnet import alexnet
import tensorflow as tf
WIDTH = 60
HEIGHT = 60
LR = 1e-3
EPOCHS = 5
MODEL_NAME = 'pyTetris2.0-epochs.model'.format(LR, 'alexnet', EPOCHS)


def train_model():
    train_data = np.load('final_data.npy')
    model = alexnet(WIDTH, HEIGHT, LR)
    train = train_data
    X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
    Y = [i[1] for i in train]

    model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS,
              snapshot_step=500, show_metric=True, batch_size=64,
              snapshot_epoch=False, run_id=MODEL_NAME)

    model.save(MODEL_NAME)
