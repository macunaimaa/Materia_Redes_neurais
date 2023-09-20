import gym
import numpy as np
from gym import spaces

class CustomGridEnv(gym.Env):
    def __init__(self, grid_size):
        super(CustomGridEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # Four possible actions: Up, Down, Left, Right
        self.observation_space = spaces.Discrete((grid_size + 2) * (grid_size + 2))
        
        self.grid = np.full((grid_size + 2, grid_size + 2), -2)  # Initialize with walls
        self.grid[1:-1, 1:-1] = 0  # Set the inner grid to 0s
        
        self.obstacle = (np.random.randint(1, grid_size + 1), np.random.randint(1, grid_size + 1))  # Avoid placing obstacle on the border
        self.target = (grid_size-1, grid_size-1)
        while self.target == self.obstacle:
            self.target = (np.random.randint(1, grid_size + 1), np.random.randint(1, grid_size + 1))

        self.reward_positive = 10
        self.reward_negative = -10
        self.max_steps = 2 * (grid_size - 1)  # Maximum steps to reach the target
        
        self.agent_pos = (0, 0)
        self.steps_taken = 0

    def reset(self):
        self.agent_pos = (1, 1)
        self.steps_taken = 0
        return self._get_observation()
    
    def step(self, action):
        self.steps_taken += 1
        new_agent_pos = self._get_new_position(self.agent_pos, action)
        
        if new_agent_pos == self.target:
            reward = self.reward_positive
            done = True
        elif new_agent_pos == self.obstacle:
            reward = self.reward_negative
            done = True
        elif self.grid[new_agent_pos] == -2:
            reward = self.reward_negative
            done = True
        elif self.steps_taken >= self.max_steps:
            reward = 0
            self.steps_taken = 0
            done = True
        else:
            reward = 0
            done = False
        
        self.agent_pos = new_agent_pos
        return self._get_observation(), reward, done, {}
    
    def render(self, mode='human'):
        for i in range(self.grid_size+1):
            for j in range(self.grid_size+1):
                if (i, j) == self.agent_pos:
                    print("A", end=' ')
                elif (i, j) == self.target:
                    print("T", end=' ')
                elif (i, j) == self.obstacle:
                    print("X", end=' ')
                else:
                    print(".", end=' ')
            print()
    
    def _get_new_position(self, position, action):
        new_pos = np.array(position)
        if action == 0:  # Up
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1:  # Down
            new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)
        elif action == 2:  # Left
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 3:  # Right
            new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
        return tuple(new_pos)
    
    def _get_observation(self):
        self.agent_pos = tuple(self.agent_pos)
        return self.agent_pos
    
    def observe_surroundings(env, state):
        surroundings = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                new_pos = (state[0] + i, state[1] + j)
                if new_pos == env.agent_pos:
                    surroundings.append(1)  # Agent's position
                elif new_pos == env.target:
                    surroundings.append(2)  # Target's position
                elif new_pos == env.obstacle:
                    surroundings.append(-1)  # Obstacle's position
                else:
                    surroundings.append(0)  # Empty space
        return surroundings