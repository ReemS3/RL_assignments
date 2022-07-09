import gym
import numpy as np
import tensorflow as tf
from actor import Actor
from critic import Critic
from tensorflow.keras.layers import Conv2D, Dense, InputLayer


class A2C():
    def __init__(self, env, gamma, alpha, lam):
        self.actor = Actor(env)
        self.critic = Critic(env)
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha

    def calculate_advantages(self, TD_errors):
        result = np.empty_like(TD_errors)
        result[-1] = TD_errors[-1]
        for t in range(len(TD_errors)-2, -1, -1):
            result[t] = TD_errors[t] + self.gamma*self.lam*result[t+1]
        return result

    def train_critic(self, trajectories, gae=False):
        for i, trajectory in enumerate(trajectories):
            episode_reward = sum([result[2] for result in trajectory])
            print("episode: ", i, "episode reward: ", episode_reward)
            rewards_t = [result[2] for result in trajectory]
            states = [result[0] for result in trajectory]          
            discounted_returns = [reward*self.gamma**(i) for i,reward in enumerate(rewards_t)]
            critic_estimation = []

            with tf.GradientTape() as tape:
                prediction = self.critic(tf.convert_to_tensor(states))
                if gae:
                    critic_estimation.append(prediction)
                loss = self.critic.loss_function(discounted_returns, prediction)
            # get gradients of this tape
            gradients = tape.gradient(loss, self.critiv.trainable_variables)
            # accumulate the gradients
            accum_gradient = [(acum_grad+grad) for acum_grad, grad in 
                           zip(accum_gradient, gradients)]

        accum_gradient = [this_grad/len(trajectories) for this_grad in accum_gradient]
        self.critic.optimizer.apply_gradients(zip(accum_gradient,self.critic.trainable_variables))
        if gae:
            TD_errors = rewards_t + self.gamma**critic_estimation[1:] - critic_estimation[:-1]
            advantages = self.calculate_advantages(TD_errors)
            # normalize advantages
            advantages = (advantages - advantages.mean())/advantages.std()
            return advantages
        return None

    def train_actor(self, trajectories, advantages=None):
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
            # get gradients of this tape
            gradients = tape.gradient(loss, self.actor.trainable_variables)
            # accumulate the gradients
            accum_gradient = [(acum_grad+grad) for acum_grad, grad in 
                           zip(accum_gradient, gradients)]

        accum_gradient = [this_grad/len(trajectories) for this_grad in accum_gradient]
        self.actor.optimizer.apply_gradients(zip(accum_gradient,self.actor.trainable_variables))

    def train(self, stepsize, render):
        trajectories = []
        for __ in range(stepsize):
            trajectory = self.actor.sample_trajectoy(render)
            trajectories.append(trajectory)
        advantages = self.train_critic(trajectories)
        if advantages:
            self.train_actor(trajectories, advantages)
        else:
            self.train_actor(trajectories, advantages)

def main():
    env = gym.make("CarRacing-v1")
    a2c = A2C(env.unwrapped, 0.99, 0.1,0.99)
    a2c.train(1000, True)
    # env.render()

if __name__ == "__main__":
    main()