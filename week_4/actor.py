import gym
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Conv2D, Dense, InputLayer, GlobalAveragePooling2D


class Actor(tf.keras.Model):
    def __init__(self, env):
        super(Actor, self).__init__()
        self.act_dim = env.action_space.shape[0]
        self.mu = tf.Variable((0.5 * np.ones(env.action_space.shape, dtype=np.float32)))
        self.optimizer = tf.optimizers.Adam(learning_rate=0.01)
        self.env = env

        self.input_layer = InputLayer(input_shape=env.observation_space.shape[0])
        self.layers_ls = [
            Conv2D(32, kernel_size=3, activation="relu"),
            Conv2D(16, kernel_size=3,activation="relu"),
            GlobalAveragePooling2D(),
            Dense(self.act_dim, activation="softmax")
        ]
    
    @tf.function
    def call(self, input, action=None):
        input = tf.expand_dims(input, axis=-1)
        print("Input ", input)
        mean = self.input_layer(input)
        for layer in self.layers_ls:
            mean = layer(mean)
        print("here", self.act_dim, mean)
        mu = tf.math.exp(self.mu)
        pi = tfp.distributions.Normal(mean, mu)
        if action is None:
            action = pi.sample(sample_shape=1)
            print(action.shape, self.act_dim)
            return action, pi.log_prob(value=action)
        return pi.log_prob(value=action)


    def sample_trajectoy(self, render, num_steps=0):
        obs = self.env.reset()

        trajectory = []
        if num_steps == 0:
            if render:
                self.env.render()
            sampled_action, log_prob = self(tf.convert_to_tensor(obs, dtype=tf.float32))
            # print(sampled_action.shape, sampled_action[0].shape )
            next_obs, rew_t, done, _ = self.env.step(sampled_action[0].numpy())
            trajectory.append((obs, sampled_action, rew_t, next_obs, log_prob, done))
            while not done:
                if render:
                    self.env.render()
                obs = next_obs
                # obs = self.process_state_image(obs)
                sampled_action, log_prob = self(obs)
                # print(sampled_action[0][0])
                next_obs, rew_t, done, _ = self.env.step(sampled_action[0].numpy())
                trajectory.append((obs, sampled_action, rew_t, next_obs, log_prob, done))
        else:
            while num_steps!=0 and not done:
                if render:
                    self.env.render()
                sampled_action, log_prob = self(obs)
                next_obs, rew_t, done, _ = self.env.step(sampled_action)
                trajectory.append((obs, sampled_action, rew_t, next_obs, log_prob, done))
                obs = next_obs
                # obs = self.process_state_image(obs)
                num_steps -= 1
        return trajectory


def main():
    env = gym.make("CarRacing-v1")
    env.reset()
    env.render()
    # env.step(0)
    # env.render()

if __name__ == "__main__":
    main()