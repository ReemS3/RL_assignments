import gym
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import keras
from tensorflow.keras.layers import Conv2D, Dense, InputLayer, GlobalAveragePooling2D, Flatten, Reshape


class Actor(tf.keras.Model):
    """
    this class is an implementation of the Actor network in the Actor-Critic architecture network
    """
    def __init__(self, env):
        super(Actor, self).__init__()
        self.act_dim = env.action_space.shape[0]
        # At the start all actions have a probability of 0.5.
        self.stddev = tf.Variable((0.5 * np.ones(env.action_space.shape, dtype=np.float32)))
        self.optimizer = tf.optimizers.Adam(learning_rate=0.01)
        self.env = env
        self.input_layer = InputLayer(input_shape=env.observation_space.shape)
        self.layers_ls = [
            Conv2D(16, kernel_size=2, activation="tanh"),
            Conv2D(8, kernel_size=2, activation= "relu"),
            GlobalAveragePooling2D(),
            Dense(self.act_dim,activation="softmax")
        ]
    
    @tf.function
    def call(self, input, action=None):
        input = tf.cast(tf.expand_dims(input, axis=0), dtype=tf.float32)
        mean = self.input_layer(input)
        for layer in self.layers_ls:
            mean = layer(mean)
        stddev = tf.math.exp(self.stddev)
        pi = tfp.distributions.Normal(mean, stddev)
        if action is None:
            action = pi.sample(sample_shape=[1])
            return action[0][0], pi.log_prob(value=action)
        return pi.log_prob(value=action)


    def sample_trajectoy(self, render, n=0):
        """Sample either a full tracjectory or only till the n-th step """
        obs = self.env.reset()
        trajectory = []
        if n == 0:
            if render:
                self.env.render()
            sampled_action, log_prob = self(obs)
            next_obs, rew_t, done, _ = self.env.step(sampled_action.numpy())
            trajectory.append((obs, sampled_action, rew_t, next_obs, log_prob, done))
            while not done:
                if render:
                    self.env.render()
                obs = next_obs
                sampled_action, log_prob = self(obs)
                next_obs, rew_t, done, _ = self.env.step(sampled_action.numpy())
                trajectory.append((obs, sampled_action, rew_t, next_obs, log_prob, done))
        else:
            done = False
            while n != 0 and not done:
                if render:
                    self.env.render()
                sampled_action, log_prob = self(obs)
                next_obs, rew_t, done, _ = self.env.step(sampled_action.numpy())
                trajectory.append((obs, sampled_action, rew_t, next_obs, log_prob, done))
                obs = next_obs
                n -= 1
        return trajectory