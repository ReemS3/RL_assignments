import gym
import numpy as np
import tensorflow as tf
from actor import Actor
from critic import Critic
from tensorflow.keras.layers import Conv2D, Dense, InputLayer


class A2C():
    def __init__(self, env, gamma, alpha, lam):
        """Implementing A2C together with GAE"""
        self.actor = Actor(env)
        self.critic = Critic(env)
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha

    def calculate_advantages(self, TD_errors):
        """credits to https://avandekleut.github.io/a2c/"""
        result = np.empty_like(TD_errors)
        result[-1] = TD_errors[-1]
        for t in range(len(TD_errors)-2, -1, -1):
            result[t] = TD_errors[t] + self.gamma*self.lam*result[t+1]
        return result

    def calculate_returns(self, rewards, dones):
        """credits to https://avandekleut.github.io/a2c/"""
        result = np.empty_like(rewards)
        result[-1] = rewards[-1]
        for t in range(len(rewards)-2, -1, -1):
            result[t] = rewards[t] + self.gamma*(1-dones[t])*result[t+1]
        return result

    def train_critic(self, trajectories, gae=False):
        """Train the critic by improving its evaluation of the sampled trajectories"""
        for __, trajectory in enumerate(trajectories):
            rewards_t = [result[2] for result in trajectory]
            states = [result[0] for result in trajectory]
            dones = [result[5] for result in trajectory]   

            critic_estimation = []
            value_estimation_laststate = self.critic(states[-1])[0]
            rewards_t[-1] = self.gamma*(1-dones[-1])*value_estimation_laststate
            discounted_returns = self.calculate_returns(rewards_t, dones)

            accum_gradient = [tf.zeros_like(tv) for tv in self.critic.trainable_variables]
            with tf.GradientTape() as tape:
                prediction = [self.critic(state)[0].numpy() for state in states]
                loss = self.critic.loss_function(discounted_returns, prediction[:-1])
            gradients = tape.gradient(loss, self.critic.trainable_variables)
            accum_gradient = [(acum_grad+grad) for acum_grad, grad in 
                           zip(accum_gradient, gradients)]

        accum_gradient = [this_grad/len(trajectories) for this_grad in accum_gradient]
        self.critic.optimizer.apply_gradients(zip(accum_gradient,self.critic.trainable_variables))
        if gae:
            prediction = self.critic(states)[0]
            if len(critic_estimation) > 1:
                TD_errors = rewards_t + self.gamma*critic_estimation[1:] - critic_estimation[:-1]
            else:
                print(rewards_t, self.gamma, critic_estimation)
                TD_errors = rewards_t + self.gamma*critic_estimation
            advantages = self.calculate_advantages(TD_errors)
            advantages = (advantages - advantages.mean())/advantages.std()
            return advantages
        return None

    def train_actor(self, trajectories, advantages=None):
        """Train the actor based on the advantages if gae is used"""
        # accum_gradients = [tf.zeros_like(trainable_variables) for trainable_variables in self.actor.trainable_variables]
        for i, trajectory in enumerate(trajectories):
            episode_reward = sum([result[2] for result in trajectory])
            print("episode: ", i, "episode reward: ", episode_reward)
            rewards_t = [result[2] for result in trajectory]
            states = [result[0] for result in trajectory]
            actions = [result[1] for result in trajectory]

            discounted_returns = [reward*self.gamma**(i) for i,reward in enumerate(rewards_t)]
            with tf.GradientTape() as tape:
                if advantages:
                    log_pro = self.actor(tf.convert_to_tensor(states),tf.convert_to_tensor(advantages))
                else:
                    log_pro = self.actor(tf.convert_to_tensor(states),tf.convert_to_tensor(actions))
                loss = tf.reduce_sum(-discounted_returns*log_pro)

            gradients = tape.gradient(loss, self.actor.trainable_variables)
            accum_gradient = [(acum_grad+grad) for acum_grad, grad in 
                           zip(accum_gradient, gradients)]

        accum_gradient = [this_grad/len(trajectories) for this_grad in accum_gradient]
        self.actor.optimizer.apply_gradients(zip(accum_gradient,self.actor.trainable_variables))

    def train(self, n, render):
        trajectories = []
        for __ in range(n):
            trajectory = self.actor.sample_trajectoy(render)
            trajectories.append(trajectory)
            advantages = self.train_critic(trajectories, gae=True)
            self.train_actor(trajectories, advantages)

def main():
    env = gym.make("CarRacing-v1")
    a2c = A2C(env.unwrapped, 0.99, 0.1, 0.95)
    a2c.train(1000, True)

if __name__ == "__main__":
    main()