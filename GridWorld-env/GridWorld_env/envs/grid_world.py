import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class GridWorldEnv(gym.Env): 
    
    def __init__(self, dimension_size):
        self.dimension_size = dimension_size
        self.timestep_elapsed = 0
        self.reset()
    
    def reset(self, seed=None, options=None):
        self.building_zone = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        
        random_start_pos = np.random.randint(0, self.dimension_size, 2)
        self.agent_pos = [random_start_pos[0], random_start_pos[1]]
        
        # 0: up, 1: down, 2: left, 3: right, 4: pick, 5: drop
        self.action_space = spaces.Discrete(6)   
        self._init_target()

        self.timestep_elapsed = 0

    def _get_obs(self):
        pass
    def _get_info(self):
        pass

    def _init_target(self):
        self.target = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        
        # Hardcoding 4 columns, and 4 beams across the 4 columns. TO BE CHANGED TO BE MORE DYNAMIC AND USEABLE
        
        points = [
            [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], # column 1, 4 block high in the left upper corner
            [0, self.dimension_size - 1, 0], [0, self.dimension_size - 1, 1], [0, self.dimension_size - 1, 2], [0, self.dimension_size - 1, 3], # column 2, 4 block high in the left lower corner
            [self.dimension_size - 1, 0, 0], [self.dimension_size - 1, 0, 1], [self.dimension_size - 1, 0, 2], [self.dimension_size - 1, 0, 3], # column 3, 4 block high in the right upper corner
            [self.dimension_size - 1, self.dimension_size - 1, 0], [self.dimension_size - 1, self.dimension_size - 1, 1], [self.dimension_size - 1, self.dimension_size - 1, 2], [self.dimension_size - 1, self.dimension_size - 1, 3], # column 4, 4 block high in the right lower corner
        ]
        
        for p in points:
            self.target[p[0], p[1], p[2]] = 1

    def step(self, action):
        self.timestep_elapsed += 1
        # 0: up, 1: down, 2: left, 3: right, 4: pick, 5: drop
        if action == 0:
            # Check if agent is at top of the grid
            if self.agent_pos[0] != 0:
                self.agent_pos[0] -= 1
        elif action == 1:
            # Check if agent is at the bottom of the grid
            if self.agent_pos[0] != self.dimension_size - 1:
                self.agent_pos[0] += 1
        elif action == 2:
            # Check if agent is at the left of the grid
            if self.agent_pos[1] != 0:
                self.agent_pos[1] -= 1
        elif action == 3:
            # Check if agent is at the right of the grid
            if self.agent_pos[1] != self.dimension_size - 1:
                self.agent_pos[1] += 1
        elif action == 4:
            # Check if there is a block to pick up. If there is a block, remove it from the building zone, else nothing happens
           if len(len(np.where(self.building_zone[self.agent_pos[0], self.agent_pos[1], :])[0]) != 0):
               removed_block_z = np.where(self.building_zone[self.agent_pos[0], self.agent_pos[1], :])[0][-1]
               self.building_zone[self.agent_pos[0], self.agent_pos[1], removed_block_z] = 0
        elif action == 5:
            # 4 neghbouring directions
            neighbour_direction = [[self.agent_pos[0] + delta_x, self.agent_pos[1] + delta_y] for delta_x, delta_y in [[-1, 0], [1, 0], [0, -1], [0, 1]]]
            for i in range(4):
                if neighbour_direction[i][0] < 0 or neighbour_direction[i][0] >= self.dimension_size or neighbour_direction[i][1] < 0 or neighbour_direction[i][1] >= self.dimension_size:
                    neighbour_direction.pop(i)
            
            # Find the highest block in all neighbouring directions
            highest_z = -1000
            for i in range(len(neighbour_direction)):
                if len(np.where(self.building_zone[neighbour_direction[i][0], neighbour_direction[i][1], :])[0]) == 0:
                    value = 0
                else:
                    value = np.max(np.where(self.building_zone[neighbour_direction[i][0], neighbour_direction[i][1], :])[0])
                    
                if value > highest_z:
                    highest_z = value
            
            # Drop the block at the highest z
            self.building_zone[self.agent_pos[0], self.agent_pos[1], highest_z] = 1
        
        # return observation, reward, terminated, truncated, info
        
        if self.timestep_elapsed >= 150:
            return (self.building_zone, self.agent_pos), -1, False, True, {}
        # Check if the building zone is the same as the target
        if np.array_equal(self.building_zone, self.target):
            return (self.building_zone, self.agent_pos), 0, True, False, {}
        
        return (self.building_zone, self.agent_pos), -1, False, False, {}
                
 
    def render(self):

        # prepare some coordinates
        cube1 = self.target == 1
        #print(cube1)

        # set the colors of each object
        colors = np.empty(cube1.shape, dtype=object)
        colors[cube1] = 'blue'
        #print(colors)

        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(cube1, facecolors=colors, edgecolor='k')

        plt.show()
        return
    
    
    def close(self):
        pass
    
if __name__ == "__main__":
    env = GridWorldEnv(4)
    #print(env.target)
    env.step(0)
