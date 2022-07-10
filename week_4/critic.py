import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Conv2D, Dense, InputLayer, Flatten


class Critic(tf.keras.Model):
    """
        this class is an implementation of the Critic network in the Actor-Critic architecture network
    """
    def __init__(self, env):
        super(Critic, self).__init__()
        self.obs_dim = env.observation_space.shape
        self.input_layer = InputLayer(input_shape=self.obs_dim)
        self.layers_ls = [
            Conv2D(16, kernel_size=2, activation="tanh"),
            Conv2D(8, kernel_size=2, activation="relu"),
            Flatten(),
            Dense(1)
        ]
        self.optimizer = tf.optimizers.Adam(learning_rate=0.01)
        self.loss_function = tf.keras.losses.MeanSquaredError()
    
    @tf.function
    def call(self, input):
        input = tf.cast(tf.expand_dims(input, axis=0), dtype=tf.float32)
        output = self.input_layer(input)
        for layer in self.layers_ls:
            output = layer(output)
        return output
