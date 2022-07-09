import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Conv2D, Dense, InputLayer


class Critic(tf.keras.Model):
    def __init__(self, env):
        super(Critic, self).__init__()
        self.obs_dim = self.adjust_according_to_space(env.observation_space)
        self.input_layer = InputLayer(input_shape=[self.obs_dim])
        self.layers_ls = [
            Conv2D(16, kernel_size=2, activation="relu"),
            Conv2D(32, kernel_size=2, activation="relu"),
            Dense(1)
        ]
        self.optimizer = tf.optimizers.Adam(learning_rate=0.01)
        self.loss_function = tf.keras.losses.MeanSquaredError()
    
    @tf.function
    def call(self, input):
        output = self.input_layer(input)
        for layer in self.layers_ls:
            output = layer(output)
        return output
        
    def adjust_according_to_space(self, env_space):
        """
            Adjusts the observation or the action space to makes it an invalid 
            input to the model networks.
        :param env_space: can be either the action or the observation space.
        :return:
        """
        if type(env_space) != gym.spaces.Box:
            dim = env_space.n
        else:
            dim = env_space.shape[0]
        return dim
