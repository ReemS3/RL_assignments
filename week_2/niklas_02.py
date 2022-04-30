from collections import defaultdict

import numpy as np
import random

AGENT = 0
BLOCKED = 1
EMPTY = 2
NEGATIVE_REWARD = 3
POSITIVE_REWARD = 4
UFO = 5
LOHRA = 6
CHARMAP = {
    AGENT: "A",
    BLOCKED: "⬛",
    EMPTY: "_",
    NEGATIVE_REWARD: "☠",
    POSITIVE_REWARD: "✌",
    UFO: "👽",
    LOHRA: "L",
}

UP = 11
DOWN = 12
LEFT = 13
RIGHT = 14
ALL_ACTIONS = [UP, DOWN, LEFT, RIGHT]
CHANGES = {UP:(-1,0), DOWN:(1,0), LEFT:(0, -1), RIGHT:(0,1) }
DIRECTIONS = {UP:"UP", DOWN:"DOWN", LEFT:"LEFT", RIGHT:"RIGHT" }

REWARD_MAP = defaultdict(lambda:0)
REWARD_MAP[NEGATIVE_REWARD] = -1
REWARD_MAP[POSITIVE_REWARD] = 71

DEFAULT_AGENT_POS = (0, 0)
DEFAULT_END_POS = (3, 2)
DEFAULT_WORLD = np.array([
    [AGENT, NEGATIVE_REWARD, LOHRA, NEGATIVE_REWARD, UFO],
    [EMPTY, EMPTY, EMPTY, EMPTY, BLOCKED],
    [EMPTY, LOHRA, BLOCKED, EMPTY, EMPTY],
    [EMPTY, BLOCKED, POSITIVE_REWARD, UFO, BLOCKED],
    [EMPTY, LOHRA, NEGATIVE_REWARD, BLOCKED, UFO],
])

class Griduniverse:
    _world: np.array = DEFAULT_WORLD.copy()
    _agent_pos: (int, int) = DEFAULT_AGENT_POS
    _local_field_value = EMPTY

    def reset(self)-> None:
        self._world = DEFAULT_WORLD.copy()
        self._agent_pos = DEFAULT_AGENT_POS
        self._local_field_value = EMPTY

    def step(self, action: int)->((int, int), int, bool):
        print(f"Executing action {DIRECTIONS[action]}")
        old_position = self._agent_pos
        self._world[self._agent_pos] = self._local_field_value
        self._agent_pos = (self._agent_pos[0] + CHANGES[action][0],self._agent_pos[1] + CHANGES[action][1])
        if self._outside_world():
            self._agent_pos = old_position
            self._local_field_value = self._world[self._agent_pos]
            self._world[self._agent_pos] = AGENT
            print("No, don't go into the bottomless sky!")
            if self._local_field_value==UFO and random.random()>0.5:
                self._world[self._agent_pos] = UFO
                self._agent_pos = self.ufo_tile_list[
                    random.randint(0, len(self.ufo_tile_list) - 1)]
                self._world[self._agent_pos] = AGENT
                print(f"You're lucky, Scottie pays attention... beam to {self._agent_pos}")
            return self._agent_pos, REWARD_MAP[self._local_field_value], self._local_field_value==POSITIVE_REWARD
        self._local_field_value = self._world[self._agent_pos]
        self._world[self._agent_pos] = AGENT
        if self._local_field_value==LOHRA:
            random_action = ALL_ACTIONS[random.randint(0, 3)]
            print(f"Partytime! You turn {DIRECTIONS[random_action]}")
            return self.step(action=random_action)
        elif self._local_field_value==UFO and random.random()>0.5:
            self._world[self._agent_pos] = UFO
            self._agent_pos = self.ufo_tile_list[random.randint(0, len(self.ufo_tile_list)-1)]
            self._world[self._agent_pos] = AGENT
            print(f"Scottie is drunk again... beam to {self._agent_pos}")
        elif self._local_field_value == BLOCKED:
            self._world[self._agent_pos] = BLOCKED
            self._agent_pos = old_position
            self._local_field_value = self._world[self._agent_pos]
            self._world[self._agent_pos] = AGENT
            print("Ups, thats a wall")
        return self._agent_pos, REWARD_MAP[self._local_field_value], self._local_field_value == POSITIVE_REWARD

    def visualize(self)->None:
        for row in range(len(self._world)):
            for column in range(len(self._world[0])):
                print(CHARMAP[self._world[(row,column)]], end=' ')
            print()
        print()

    @property
    def ufo_tile_list(self):
        return [(row, column) for row, column in zip(range(len(self._world)),range(len(self._world[0]))) if self._world[(row, column)] == UFO]

    def _outside_world(self):
        return self._agent_pos[0] < 0 or self._agent_pos[0] >= len(self._world) or self._agent_pos[1] < 0 or self._agent_pos[1] >= len(self._world[0])

    @property
    def state(self):
        return self._agent_pos

class Sams:
    _epsilon: float = 0.05
    _alpha: float = 0.05
    _gamma: float = 0.95

    def __init__(self):
        self._q_table = {}
        for row in range(len(DEFAULT_WORLD)):
            for column in range(len(DEFAULT_WORLD[0])):
                for action in ALL_ACTIONS:
                    self._q_table[((row, column), action)] = random.random() - 0.5
        # terminal state
        for action in ALL_ACTIONS:
            self._q_table[((DEFAULT_END_POS, DEFAULT_END_POS), action)] = 0

    def sample_action(self, state: (int, int))->int:
        if random.random() > self._epsilon:
            print(f"maximizing q-value for state {state}")
            return max([(self._q_table[(state,action)], action) for action in ALL_ACTIONS])[1]
        print("epsilon made mankind to explorers!")
        return ALL_ACTIONS[random.randint(0, 3)]

    def update(self, reward: int, new_state: (int, int), state: (int, int), action: int)->int:
        new_action = self.sample_action(state=new_state)
        self._q_table[(state, action)]+=self._alpha * (reward + self._gamma * self._q_table[(new_state, new_action)] -self._q_table[(state, action)])
        return new_action

random.seed(35)
my_world = Griduniverse()
my_world.visualize()
state = my_world.state
hero = Sams()

N = 1
all_rewards = []
all_lengths = []
for j in range(10000):
    my_world.reset()
    action = hero.sample_action(state=state)
    total_reward = 0
    for i in range(10000):
        print(f"episode {j} epoch {i} total reward is {total_reward}")
        new_state, reward, terminal = my_world.step(action=action)
        my_world.visualize()
        new_action = hero.update(reward=reward, state=state, action=action, new_state=new_state)
        state=new_state
        action = new_action
        total_reward += reward
        if terminal:
            print("end score:", total_reward)
            all_rewards.append(total_reward)
            all_lengths.append(i)
            break

print(all_rewards)
print(all_lengths)