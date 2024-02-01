import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

MAX_TIMESTEP = 500

class GridWorldEnv(gym.Env): 
    
    def __init__(self, dimension_size):
        self.dimension_size = dimension_size
        self.timestep_elapsed = 0
        self.reset()
    
    def reset(self, seed=None, options=None):
        self.building_zone = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        
        random_start_pos = np.random.randint(0, self.dimension_size, 3)
        self.agent_pos = [random_start_pos[0], random_start_pos[1], random_start_pos[2]]
        
        # List of actions
        # 0: forward, 1: backward
        # 2: left, 3: right
        # 4: up, 5: down
        # 6: pick
        self.action_space = spaces.Discrete(7)   
        self._init_target()

        self.timestep_elapsed = 0

    def _get_obs(self):
        #np.concatenate((self.building_zone, self.agent_pos), dim=0)
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
        move_cmd = False
        place_cmd = False
        # List of actions
        # 0: forward, 1: backward
        # 2: left, 3: right
        # 4: up, 5: down
        # 6: pick
        if action == 0:
            # Y - 1
            if self.agent_pos[1] > 0:
                self.agent_pos[1] -= 1
            move_cmd = True
        elif action == 1:
            # Y + 1
            if self.agent_pos[1] < self.dimension_size - 1:
                self.agent_pos[1] += 1
            move_cmd = True
                
        elif action == 2:
            # X - 1
            if self.agent_pos[0] > 0:
                self.agent_pos[0] -= 1
            move_cmd = True
                
        elif action == 3:
            # X + 1
            if self.agent_pos[0] < self.dimension_size - 1:
                self.agent_pos[0] += 1
            move_cmd = True
        
        elif action == 4:
            # Z + 1
            if self.agent_pos[2] < self.dimension_size - 1:
                self.agent_pos[2] += 1
            move_cmd = True       
        elif action == 5:
            # Z - 1
            if self.agent_pos[2] > 0:
                self.agent_pos[2] -= 1
            move_cmd = True
        
        elif action == 6: # Place a block
            place_cmd = True
            # Find all 6 neighbouring directions
            neighbour_direction = [
                [self.agent_pos[0] + delta_x, self.agent_pos[1] + delta_y, self.agent_pos[2] + delta_z]
                for delta_x, delta_y, delta_z in [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
            ]
            
            for neighbour in neighbour_direction:
                if neighbour[0] < 0 or neighbour[0] >= self.dimension_size or neighbour[1] < 0 or neighbour[1] >= self.dimension_size or neighbour[2] < 0 or neighbour[2] >= self.dimension_size:
                    neighbour_direction.remove(neighbour)
            
            # Find if there is any supporting neighbour
            supporting_neighbour = False
            for neighbour in neighbour_direction:
                if self.building_zone[neighbour[0], neighbour[1], neighbour[2]] == 1:
                    supporting_neighbour = True
                    break
            
            if supporting_neighbour:
                # Place the block
                self.building_zone[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] = 1
            else:
                # Check if block on the ground. No need to check support
                if self.agent_pos[2] == 0:
                    self.building_zone[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] = 1
                    supporting_neighbour = True
            
        # return observation, reward, terminated, truncated, info
        if move_cmd:
            if self.timestep_elapsed > MAX_TIMESTEP:
                return self._get_obs(), -1, False, True, {}
            else:
                return self._get_obs(), -1, False, False, {}
        elif place_cmd:
            if supporting_neighbour:
                if np.equal(self.building_zone, self.target).all():
                    return self._get_obs(), 0, True, False, {}
                else:
                    if self.timestep_elapsed > MAX_TIMESTEP:
                        return self._get_obs(), -1, False, True, {}
                    else:
                        return self._get_obs(), -1, False, False, {}
            else:
                if self.timestep_elapsed > MAX_TIMESTEP:
                    return self._get_obs(), -100, False, True, {}        
                else:
                    return self._get_obs(), -100, False, False, {}
 
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
