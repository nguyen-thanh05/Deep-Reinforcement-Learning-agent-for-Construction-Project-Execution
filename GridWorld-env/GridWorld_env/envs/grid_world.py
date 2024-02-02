import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import torch
MAX_TIMESTEP = 200

class GridWorldEnv(gym.Env): 
    
    def __init__(self, dimension_size):
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.dimension_size = dimension_size
        self.timestep_elapsed = 0
        self.reset()
    
    def reset(self, seed=None, options=None):
        self.building_zone = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        
        random_start_pos = np.zeros(3, dtype=int)
        self.agent_pos = [random_start_pos[0], random_start_pos[1], random_start_pos[2]]
        
        # List of actions
        # 0: forward, 1: backward
        # 2: left, 3: right
        # 4: up, 5: down
        # 6: pick
        self.action_space = spaces.Discrete(7)   
        self._init_target()
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)

        self.timestep_elapsed = 0
        return self.get_obs(), {}

    def get_obs(self):
        # clear agent_pos_grid
        agent_pos_grid = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        agent_pos_grid[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] = 1
        
        return np.stack((self.building_zone, agent_pos_grid, self.target), axis=0)
        
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
        if action == 1:
            # Y - 1
            if self.agent_pos[1] > 0:
                self.agent_pos[1] -= 1
            move_cmd = True
        elif action == 0:
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
        
        print(self.timestep_elapsed, MAX_TIMESTEP, self.timestep_elapsed > MAX_TIMESTEP)

        # return observation, reward, terminated, truncated, info
        if move_cmd:
            if self.timestep_elapsed > MAX_TIMESTEP:
                return self.get_obs(), torch.tensor(-1), torch.tensor(0), torch.tensor(1), {}
            else:
                return self.get_obs(), torch.tensor(-1), torch.tensor(0), torch.tensor(0), {}
        elif place_cmd:
            #self.render()
            if supporting_neighbour:
                if np.equal(self.building_zone, self.target).all():
                    return self.get_obs(), 0, torch.tensor(1), torch.tensor(0), {}
                else:
                    if self.timestep_elapsed > MAX_TIMESTEP:
                        return self.get_obs(), torch.tensor(-1), torch.tensor(0), torch.tensor(1), {}
                    else:
                        return self.get_obs(), torch.tensor(-1), torch.tensor(0), torch.tensor(0), {}
            else:
                if self.timestep_elapsed > MAX_TIMESTEP:
                    return self.get_obs(), torch.tensor(-5), torch.tensor(0), torch.tensor(1), {}        
                else:
                    return self.get_obs(), torch.tensor(-5), torch.tensor(0), torch.tensor(0), {}
 
    def render(self):
        agent_pos_grid = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        agent_pos_grid[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] = 1

        # prepare some coordinates
        targetCube = self.building_zone == 1
        agentCube = agent_pos_grid == 1




        cube1 = targetCube | agentCube
        # set the colors of each object
        colors = np.empty(cube1.shape, dtype=object)
        colors[targetCube] = 'blue'
        colors[agentCube] = 'yellow'
        #print(colors)

        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(cube1, facecolors=colors, edgecolor='k')

        plt.show()
        return
    
    
    def close(self):
        pass
    
if __name__ == "__main__":
    env = GridWorldEnv(4)
    # List of actions
    # 0: forward, 1: backward
    # 2: left, 3: right
    # 4: up, 5: down
    # 6: place block
    #env.step(0)
    #env.step(6)
    #env.step(4)
    #env.step(6)
    #env.step(3)
    #env.step(6)
    ##env.render()
    #print(env.building_zone)

    env.reset()
    for _ in range(1000):
        # sample discrete random action between 0 and 6 using numpy
        action = np.random.randint(0, 7)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            env.reset()
