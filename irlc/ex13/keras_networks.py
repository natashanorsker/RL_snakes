"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
# https://medium.com/@leosimmons/double-dqn-implementation-to-solve-openai-gyms-cartpole-v-0-df554cd0614d
import tensorflow as tf
# tf.keras.backend.set_floatx('float64')
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import he_normal
from irlc.ex13.dqn_network import DQNNetwork
import numpy as np
import os

class KerasNetwork(DQNNetwork):
    def __init__(self, env, trainable=True, learning_rate=0.001):
        super().__init__()
        self.env = env
        self.model = self.build_model_()
        if trainable:
            self.model.compile(optimizer=Adam(lr=learning_rate), loss=tf.keras.losses.Huber()) # 'mse'

    def build_model_(self):
        num_actions = self.env.action_space.n
        input, dense2, = build_cartpole(self.env, hidden_size=30)
        adv_dense = Dense(num_actions)(dense2)
        return keras.Model(inputs=input, outputs=adv_dense)

    def __call__(self, s):
        return self.model.predict(s)

    def fit(self, s, target):
        self.model.train_on_batch(s, target)

    def update_Phi(self, source, tau=None):
        """
        Polyak adapt weights of this class given source:
        I.e. tau=1 means adopt weights in one step,
        tau = 0.001 means adopt very slowly.
        """
        if tau is None:  # Do full overwrite of weights
            tau = 1
        weights = [ wa*(1 - tau) + wb * tau for wa, wb in zip(self.model.get_weights(),  source.model.get_weights())]
        self.model.set_weights(weights)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        self.model.save_weights(path + ".h5")

    def load(self, path):
        self.model.load_weights(path + ".h5")

def build_cartpole(env, hidden_size):
    num_observations = np.prod(env.observation_space.shape)
    input = keras.Input(shape=(num_observations,), name="input")
    dense1 = layers.Dense(hidden_size, activation='relu', kernel_initializer=he_normal())(input)
    dense2 = layers.Dense(hidden_size, activation='relu', kernel_initializer=he_normal())(dense1)
    return input, dense2

def build_duel(dense2, num_actions, hidden_size, init): 
    """
    Return tensor corresponding to Q-values when using dueling Q-networks (see exercise description)
    """
    # TODO: 7 lines missing.
    raise NotImplementedError("Implement function body")

class KerasDuelNetwork(KerasNetwork):
    def build_model_(self):
        num_actions = self.env.action_space.n
        hidden_size = 30
        input, dense2, = build_cartpole(self.env, hidden_size=hidden_size)
        combine = build_duel(dense2, num_actions=num_actions, hidden_size=hidden_size, init=he_normal)
        return keras.Model(inputs=input, outputs=combine)

class KerasDuelNetworkAtari(KerasNetwork):
    def build_model_(self):
        hidden_size = 256
        ospace = self.env.observation_space.shape
        num_actions = self.env.action_space.n
        input = keras.Input(shape=ospace, name="input")
        conv1 = keras.layers.Conv2D(16, (8, 8), (4, 4), activation='relu')(input)
        conv2 = keras.layers.Conv2D(32, (4, 4), (2, 2), activation='relu')(conv1)
        flatten = keras.layers.Flatten()(conv2)
        combine = build_duel(flatten, num_actions=num_actions, hidden_size=hidden_size, init=he_normal)
        # combine = tf.cast(combine, tf.float32)
        # combine = Lambda(lambda x: tf.cast(x, 'float32'), name='change_to_float')(combine)

        return keras.Model(inputs=input, outputs=combine)
