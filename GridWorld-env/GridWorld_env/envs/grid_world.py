import gymnasium as gym
from gymnasium import spaces
import numpy as np


class GridWorldEnv(gym.Env): 
    
    def __init__(self, dimension_size):
        self.dimension_size = dimension_size
        self.reset()
    
    def reset(self, seed=None, options=None):
        self.building_zone = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        self.agent_position = np.zeros((self.dimension_size, self.dimension_size), dtype=int)
        self.agent_position[0, 0] = 1 # Starts in the upper left corner
        # 0: up, 1: down, 2: left, 3: right, 4: pick, 5: drop
        self.action_space = spaces.Discrete(6)   
        self.target()

    def _get_obs(self):
        pass
    def _get_info(self):
        pass

    def target(self):
        self.target = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        
        # Hardcoding 4 columns, and 4 beams across the 4 columns. TO BE CHANGED TO BE MORE DYNAMIC AND USEABLE
        
        points = [
            [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], # column 1, 4 block high in the left upper corner
            [0, self.dimension_size - 1, 0], [0, self.dimension_size - 1, 1], [0, self.dimension_size - 1, 2], [0, self.dimension_size - 1, 3], # column 2, 4 block high in the left lower corner
            [self.dimension_size - 1, 0, 0], [self.dimension_size - 1, 0, 1], [self.dimension_size - 1, 0, 2], [self.dimension_size - 1, 0, 3], # column 3, 4 block high in the right upper corner
            [self.dimension_size - 1, self.dimension_size - 1, 0], [self.dimension_size - 1, self.dimension_size - 1, 1], [self.dimension_size - 1, self.dimension_size - 1, 2], [self.dimension_size - 1, self.dimension_size - 1, 3] # column 4, 4 block high in the right lower corner
        ]
        
        for p in points:
            self.target[p[0], p[1], p[2]] = 1

    def step(self, action):
        pass

    def render(self):
        pass
    
    def _render_frame(self):
        pass
    def close(self):
        pass
    
if __name__ == "__main__":
    env = GridWorldEnv(4)
    print(env.target)