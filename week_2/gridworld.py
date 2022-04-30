import random
import numpy as np


class GridWorld:
    """Create a GridWord with SARSA/Monte Carlo to solve it."""

    def __init__(self, width=5, height=5):
        self.width, self.height = width, height
        self.tiles = np.zeros((width, height), dtype=object)
        self.negative_tiles_indices = []
        self.blocked_tiles_indices = []
        self.non_deterministic_tiles = []

        self.q_values = np.ndarray(
            (width, height, 4), buffer=np.array([[0] * 4] * (width * height))
        )

        self.terminal_tile_index = 0
        self.initial_states = []
        self.player_index = [0, 0]
        self.cumulative_rewards = 0
        self._create_grid()

    def _assign_rewards(self):
        """Assign the negative rewards and 'X' for blocked terminals."""
        for x_idx, y_idx in self.negative_tiles_indices:
            self.tiles[x_idx, y_idx] = round(random.uniform(-2, -1), 2)
        for x_idx, y_idx in self.blocked_tiles_indices:
            self.tiles[x_idx, y_idx] = "X"
        self.tiles[self.terminal_tile_index[0], self.terminal_tile_index[1]] = 1.0

    def _save_initial_state(self):
        """Save the initial states of the grid."""
        self.initial_states = np.copy(self.tiles)

    def _shuffle_indices(self, rows, columns):
        """Shuffle the row and column indices."""
        random.shuffle(rows)
        random.shuffle(columns)

    def _create_blocks(self, row_indices, column_indices, number_of_blocks):
        """Create the block with indices for the specified number of blocks.

        Parameters:
        row_indices (array of int): the row indices
        column_indices (array of int): the column indices

        Returns:
        array of tuple of int

        """

        blocks = [
            idx
            for i in range(number_of_blocks)
            if (idx := (row_indices[i], column_indices[i])) != tuple(self.player_index)
        ]
        return blocks

    def _create_grid(self):
        """Create the grid and specify the negative, blocked blocks and terminal block."""
        row_indices = random.sample(range(0, self.width), self.width)
        column_indices = random.sample(range(0, self.height), self.height)
        min_range = min(self.width, self.height)

        self._shuffle_indices(row_indices, column_indices)
        self.negative_tiles_indices = self._create_blocks(
            row_indices, column_indices, min_range - 2
        )

        self._shuffle_indices(row_indices, column_indices)
        self.blocked_tiles_indices = self._create_blocks(
            row_indices, column_indices, min_range - 1
        )

        self._shuffle_indices(row_indices, column_indices)
        self.non_deterministic_tiles = self._create_blocks(
            row_indices, column_indices, min_range - 1
        )

        while True:
            terminal_idx = (
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1),
            )
            if terminal_idx not in self.negative_tiles_indices:
                if terminal_idx not in self.blocked_tiles_indices:
                    if terminal_idx != tuple(self.player_index):
                        break

        self.terminal_tile_index = terminal_idx
        self._assign_rewards()
        self._save_initial_state()

    def reset(self):
        """Reset the grid."""
        self.tiles = np.copy(self.initial_states)
        self.player_index = [0, 0]
        self.cumulative_rewards = 0.0

    def _check_if_blocked(self):
        return tuple(self.player_index) not in self.blocked_tiles_indices

    def _check_if_terminal(self):
        return tuple(self.player_index) == self.terminal_tile_index

    def step(self, action):
        """Step in the grid while making sure if the action is allowed.

        Parameters:
        action (int): the index of the chosen index

        Returns:
        new_state (tuple of int): the new state
        reward_t (float): the reward to go
        is_terminal (bool): check if the agent reache a final state

        """

        new_state, reward_t, is_terminal = None, 0, False

        # down
        if action == 0 and self.player_index[0] < 4:
            tmp = self.player_index[0]
            self.player_index[0] += 1

            if self.player_index in self.non_deterministic_tiles:
                self.player_index[0] -= 1
            if self._check_if_blocked():
                reward_t = self.tiles[self.player_index[0], self.player_index[1]]
                if self._check_if_terminal():
                    is_terminal = True
            else:
                self.player_index[0] = tmp
        # up
        elif action == 1 and self.player_index[0] > 0:
            tmp = self.player_index[0]
            self.player_index[0] -= 1
            if self._check_if_blocked():
                reward_t = self.tiles[self.player_index[0], self.player_index[1]]
                if self._check_if_terminal():
                    is_terminal = True
            else:
                self.player_index[0] = tmp

        # right
        elif action == 2 and self.player_index[1] < 4:
            tmp = self.player_index[1]
            self.player_index[1] += 1
            if self._check_if_blocked():
                reward_t = self.tiles[self.player_index[0], self.player_index[1]]
                if self._check_if_terminal():
                    is_terminal = True
            else:
                self.player_index[1] = tmp

        # left
        elif action == 3 and self.player_index[1] > 0:
            tmp = self.player_index[1]
            self.player_index[1] -= 1
            if self._check_if_blocked():
                reward_t = self.tiles[self.player_index[0], self.player_index[1]]
                if self._check_if_terminal():
                    is_terminal = True
            else:
                self.player_index[1] = tmp

        new_state = self.player_index

        return new_state, reward_t, is_terminal

    def visualise(self):
        """Visualises the current state of the grid."""
        width = (8 * "-") * self.height

        player_index = ""
        for x in range(self.width):
            print(width)
            row = []
            for y in range(self.height):
                if (x, y) == tuple(self.player_index):
                    player_index = "P"
                else:
                    player_index = " "
                if len(str(self.tiles[x][y])) == 1:
                    cell = f"|{player_index}  " + str(self.tiles[x][y]) + "   "
                elif len(str(self.tiles[x][y])) == 3:
                    cell = f"|{player_index} " + str(self.tiles[x][y]) + "  "
                elif len(str(self.tiles[x][y])) == 4:
                    cell = f"|{player_index}" + str(self.tiles[x][y]) + "  "
                else:
                    cell = f"|{player_index}" + str(self.tiles[x][y]) + " "
                row.append(cell)
            row.append("|")
            row = "".join(row)
            print(row)
        print(width)

    def choose_action(self, state, eps):
        """Choose the action either greedily or with a small probability randomly.

        Parameters:
        state (tuple of int): the current state
        ep (float): epsilon for epsilon-greedy

        Returns:
        action (int) : index of the choosen action

        """

        stochastic_probability = random.uniform(0, 1)
        if stochastic_probability < (eps / 4):
            return np.random.randint(4)
        rewards = self.q_values[state[0], state[1]]
        return np.argmax(rewards)

    def update(self, state_t, state_t1, reward, action_t, action_t1, gamma):
        """Update estimation Q for the current state.

        Parameters:
        state_t (tuple of int): the current state
        state_t1 (tuple of int): the next state
        reward (float): the returns
        action_t (int): the chosen performed action at state_t
        action_t1 (int): the action performed at state_t1
        gamma (float): the discounted factor

        """

        alpha = 0.2
        Q_t = self.q_values[state_t[0], state_t[1]][action_t]
        Q_t1 = self.q_values[state_t1[0], state_t1[1]][action_t1]
        target = reward + gamma * Q_t1

        self.q_values[state_t[0], state_t[1]][action_t] = Q_t + alpha * (target - Q_t)

    def solve(self, n_steps=1, episodes=50000):
        """Solve the problem using SARSA or Monte Carlo.

        Parameters:
        n_steps (int): number of sampling steps before doing an estimation for the next state.
        episodes (int): number of episodes to sample

        """

        actions = ["down", "up", "right", "left"]
        eps = 0.5
        eps_decay = 0.00005
        gamma = 0.99

        for _ in range(episodes):
            # decay the epsilon value until it reaches the threshold of 0.01
            if eps > 0.01:
                eps -= eps_decay

            action_t = self.choose_action(self.player_index, eps)
            state_t = list(self.player_index)
            q_value_before = self.q_values[state_t[0], state_t[1]][action_t]
            returns = 0
            t = 0
            while t < n_steps or n_steps == 0:
                # try all actions
                self.visualise()
                print("Performed Action:", actions[action_t])
                sampling_results = self.step(action_t)
                self.visualise()

                action_t1 = self.choose_action(self.player_index, eps)
                state_t1, reward, terminal = sampling_results
                action_t = action_t1
                returns += reward * (gamma**t)
                t += 1

                if terminal:
                    self.reset()
                    break
            # SARSA update
            if n_steps > 0:
                self.update(
                    state_t,
                    state_t1,
                    returns,
                    action_t,
                    action_t1,
                    gamma**n_steps,
                )
            # MC estimation
            else:
                self.q_values[state_t[0], state_t[1]][action_t] = np.average(returns)
            q_value_after = self.q_values[state_t[0], state_t[1]][action_t]
            print(
                f"Update Q-Value at state {state_t} from {q_value_before} to {q_value_after}"
            )
            if terminal:
                break


def main():
    """Just run this script to see the grid."""
    x = GridWorld(5, 5)
    # if you want to MC choose the number of steps == 0
    x.solve(n_steps=1, episodes=500000)


if __name__ == "__main__":
    main()
