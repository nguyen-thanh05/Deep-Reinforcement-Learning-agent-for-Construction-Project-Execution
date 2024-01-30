import gymnasium as gym
from gymnasium import spaces
import numpy as np


class GridWorldEnv(gym.Env): 
    
    def __init__(self, dimension_size):
        self.reset()
    
    def reset(self, seed=None, options=None):
        self.building_zone = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        self.agent_position = np.zeros((self.dimension_size, self.dimension_size), dtype=int)
        
        # 0: up, 1: down, 2: left, 3: right, 4: pick, 5: drop
        self.action_space = spaces.Discrete(6)   
        

    def _get_obs(self):
        pass
    def _get_info(self):
        pass

    

    def step(self, action):
        pass

    def render(self):
        pass
    
    def _render_frame(self):
        pass
    def close(self):
        pass
    
if __name__ == "__main__":
    env = GridWorldEnv(32)