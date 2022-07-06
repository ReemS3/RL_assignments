from neural_network import NeuralNetwork
from experience_buffer import ExperienceBuffer
from collections import deque
from gym import wrappers
import wandb
import tensorflow as tf
import numpy as np
import gym, os


class DQN:
    def __init__(self, name_of_experiment, env, max_size_buffer):
        self.train_network = NeuralNetwork(env)
        self.target_network = NeuralNetwork(env)
        self.buffer = ExperienceBuffer(max_size_buffer)
        self.minibatch_size = 512
        self.gamma = 0.99

        # Uncomment this to conduct are experiments using W&B.
        self.init_wandb(name=name_of_experiment)

    def _get_actions(self, obs, epsilon):
        """Implements greedy policy credits to
        https://github.com/VXU1230/Medium-Tutorials/blob/master/dqn/cart_pole.py"""
        if np.random.random() < epsilon:
            return np.random.choice(4)
        else:
            return np.argmax(self.train_network(np.atleast_2d(obs))[0])

    def sample(self, env, epsilon):
        obs = env.reset()
        episode_reward = 0
        done = False
        losses = []
        while not done:
            action = self._get_actions(obs, epsilon)
            new_obs, reward_t, done, _ = env.step(action)
            episode_reward += reward_t
            obs = new_obs
            self.buffer.push((obs, action, reward_t, new_obs, done))
            env.render()
            if done:
                obs = env.reset()
            loss = self.train()
            losses.append(loss)
        return episode_reward, np.mean(losses)

    def train(self):
        """
        Train the train network and update its paramteres. credits to

        https://towardsdatascience.com/deep-reinforcement-learning-build-
        a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-
        8e105744b998
        """
        minibatch_size = (
            self.minibatch_size
            if self.buffer.get_length() > self.minibatch_size
            else self.buffer.get_length()
        )
        ids = np.random.randint(
            low=0, high=self.buffer.get_length(), size=minibatch_size
        )
        states = np.asarray([self.buffer.buffer[i][0] for i in ids])
        actions = np.asarray([self.buffer.buffer[i][1] for i in ids])
        rewards = np.asarray([self.buffer.buffer[i][2] for i in ids])
        states_next = np.asarray([self.buffer.buffer[i][3] for i in ids])
        dones = np.asarray([self.buffer.buffer[i][4] for i in ids])

        value_next = np.max(self.target_network(states_next), axis=1)

        actual_action_values = np.where(
            dones, rewards, rewards + self.gamma * value_next
        )

        # improve the prediction of the network
        with tf.GradientTape() as tape:
            predicted_action_values = tf.math.reduce_sum(
                self.train_network(states) * tf.one_hot(actions, 4), axis=1
            )
            loss = tf.math.reduce_mean(
                tf.square(actual_action_values - predicted_action_values)
            )
        variables = self.train_network.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.train_network.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def target_soft_update(self, net, target_net, soft_tau):
        """soft update the target network using Polyak averaging.
        https://www.programmersought.com/article/68126496626/"""
        for target_param, param in zip(
            target_net.trainable_weights, net.trainable_weights
        ):
            target_param.assign(
                target_param * (1.0 - soft_tau) + param * soft_tau
            )
        return target_net

    def init_wandb(self, name):
        """
            Initialise a Weights&Bias project to save the runs.
        :param name: name of the projects.
        :return:
        """
        self.config = wandb.init(
            project=name, entity="rfarah", sync_tensorboard=True
        ).config


def make_video(env, TrainNet):
    env = wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)
    rewards = 0
    steps = 0
    done = False
    observation = env.reset()
    while not done:
        action = TrainNet.get_action(observation, 0)
        observation, reward, done, _ = env.step(action)
        steps += 1
        rewards += reward
    print("Testing steps: {} rewards {}: ".format(steps, rewards))


def main():
    env = gym.make("LunarLander-v2")

    dqn = DQN(
        name_of_experiment="test2", env=env.unwrapped, max_size_buffer=512
    )

    epsilon, decay = 0.99, 0.9999
    min_epsilon = 0.1
    losses = []

    N = 10000
    total_rewards = np.empty(N)

    first_time = True
    while (
        dqn.target_network.trainable_weights
        != dqn.train_network.trainable_weights
        or first_time
    ):
        for n in range(N):
            epsilon = max(min_epsilon, epsilon * decay)
            episode_reward, losses = dqn.sample(env, epsilon)
            total_rewards[n] = episode_reward
            avg_rewards = total_rewards[max(0, n - 100) : (n + 1)].mean()
            wandb.log({"episode reward": episode_reward})
            wandb.log({"losses": episode_reward, "steps": n})
            if n % 100 == 0:
                print(
                    "episode:",
                    n,
                    "episode reward:",
                    episode_reward,
                    "epsilon:",
                    epsilon,
                    "avg reward (last 100):",
                    avg_rewards,
                    "episode loss: ",
                    losses,
                )
                dqn.target_soft_update(
                    dqn.train_network, dqn.target_network, 1e-2
                )

        first_time = False

    make_video(env, dqn.train_network)
    env.close()


if __name__ == "__main__":
    main()
