import gym
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer


class NeuralNetwork(tf.keras.Model):
    def __init__(self, env):
        super(NeuralNetwork, self).__init__()
        obs_dim = self.adjust_according_to_space(env.observation_space)
        act_dim = self.adjust_according_to_space(env.action_space)
        self.input_layer = InputLayer(input_shape=[obs_dim])
        self.layers_ls = [
            Dense(16, input_shape=[obs_dim], activation="relu"),
            Dense(32, activation="relu"),
            Dense(act_dim, input_shape=[16], activation="softmax"),
        ]
        self.optimizer = tf.optimizers.Adam(learning_rate=0.01)

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
