from math import ceil
from os import dup
from turtle import onclick
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import torch
MAX_TIMESTEP = 800


class GridWorldEnv(gym.Env): 
    
    def __init__(self, dimension_size):
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.dimension_size = dimension_size
        self.timestep_elapsed = 0
        self.record_sequence = []
        self.reset()
    
    def reset(self, seed=None, options=None):
        self.building_zone = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        
        random_start_pos = np.zeros(3, dtype=int)
        self.agent_pos = [random_start_pos[0], random_start_pos[1], random_start_pos[2]]
        self.record_sequence = []
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
        possible_targets = []
        
        points = []
        for i in range(self.dimension_size):
            points.append([0, 0, i]) # column 1 at position (0, 0)
            points.append([0, self.dimension_size - 1, i]) # column 2 at position (0, dimension_size - 1)
            points.append([self.dimension_size - 1, 0, i]) # column 3 at position (dimension_size - 1, 0)
            points.append([self.dimension_size - 1, self.dimension_size - 1, i]) # column 4 at position (dimension_size - 1, dimension_size - 1)
            
            points.append([0, i, self.dimension_size - 1]) # beam 1 connecting column 1 and column 2
            points.append([self.dimension_size - 1, i, self.dimension_size - 1]) # beam 2 connecting column 3 and column 4
            points.append([i, 0, self.dimension_size - 1]) # beam 3 connecting column 1 and column 3
            points.append([i, self.dimension_size - 1, self.dimension_size - 1]) # beam 4 connecting column 2 and column 4
                   
        
        possible_targets.append(points)                      
        
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
        isInsideBlock = self.building_zone[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] == 1
        #print("Current zone", self.building_zone[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]], isInsideBlock)
        #print("Action", action)
        blockBelow = self.building_zone[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2] - 1] == 1
        tried_cmd = False
        if action == 1:
            # Y - 1
            if self.agent_pos[1] > 0: 
                blockAfter = self.building_zone[self.agent_pos[0], self.agent_pos[1] - 1, self.agent_pos[2]] == 1
                if (blockAfter or isInsideBlock or self.agent_pos[2]==0):
                    self.agent_pos[1] -= 1
                else:
                    tried_cmd = True

            move_cmd = True
        elif action == 0:
            # Y + 1

            if self.agent_pos[1] < self.dimension_size - 1:
                blockAfter = self.building_zone[self.agent_pos[0], self.agent_pos[1] + 1, self.agent_pos[2]] == 1

                if (blockAfter or isInsideBlock  or self.agent_pos[2]==0):
                    self.agent_pos[1] += 1
                else: 
                    tried_cmd = True

            move_cmd = True
                
        elif action == 2:
            # X - 1

            if self.agent_pos[0] > 0:
                blockAfter = self.building_zone[self.agent_pos[0] - 1, self.agent_pos[1], self.agent_pos[2]] == 1

                if (isInsideBlock or blockAfter or self.agent_pos[2] ==0):
                    self.agent_pos[0] -= 1
                else:
                    tried_cmd = True
            move_cmd = True
                
        elif action == 3:
            # X + 1
            #isInsideBlock = self.building_zone[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] == 1


            if self.agent_pos[0] < self.dimension_size - 1: 
                blockAfter = self.building_zone[self.agent_pos[0] + 1, self.agent_pos[1], self.agent_pos[2]] == 1

                if (isInsideBlock or blockAfter or self.agent_pos[2] == 0):
                    self.agent_pos[0] += 1
                else:
                    tried_cmd = True

            move_cmd = True
        
        elif action == 4:
            # Z + 1
            #print(isInsideBlock)
            if (self.agent_pos[2] < self.dimension_size - 1) and isInsideBlock:
                self.agent_pos[2] += 1
            move_cmd = True       
        elif action == 5:
            # Z - 1
            if self.agent_pos[2] > 0 and (isInsideBlock or blockBelow):
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
            
            duplicate_block = False
            if supporting_neighbour:
                # If the space is already occupied
                if self.building_zone[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] == 1:
                    duplicate_block = True
                # Place the block
                self.building_zone[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] = 1
            else:
                # If the space is already occupied
                if self.building_zone[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] == 1:
                    duplicate_block = True
                # Check if block on the ground. No need to check support
                if self.agent_pos[2] == 0:
                    self.building_zone[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] = 1
                    supporting_neighbour = True
                
                
        

        # return observation, reward, terminated, truncated, info
        if move_cmd:

            if self.timestep_elapsed > MAX_TIMESTEP:
                return self.get_obs(), -1, False, True, {}
            else:
                if tried_cmd:
                    return self.get_obs(), -2, False, False, {}

                else:
                    return self.get_obs(), -1, False, False, {}
        elif place_cmd:
            if supporting_neighbour and not duplicate_block:
                difference = self.target - self.building_zone
                difference = np.isin(difference, 1)
                if np.any(difference) == False:
                    return self.get_obs(), 0, True, False, {}
                else:
                    if self.building_zone[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] == self.target[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]]:
                        if self.timestep_elapsed > MAX_TIMESTEP:
                            return self.get_obs(), 1, False, True, {}
                        else:
                            return self.get_obs(), 1, False, False, {}
                    else:
                        if self.timestep_elapsed > MAX_TIMESTEP:
                            return self.get_obs(), -1.5, False, True, {}
                        else:
                            return self.get_obs(), -1.5, False, False, {}
            elif duplicate_block or not supporting_neighbour:
                if self.timestep_elapsed > MAX_TIMESTEP:
                    return self.get_obs(), -3.5, False, True, {}        
                else:
                    return self.get_obs(), -3.5, False, False, {}
 
    
    def render(self):
        agent_pos_grid = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        agent_pos_grid[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] = 1

        # prepare some coordinates
        building_zone_cube = self.building_zone == 1
        agent_position_cube = agent_pos_grid == 1


        fig = plt.figure()

        final_rendering_cube = building_zone_cube | agent_position_cube
        # set the colors of each object
        colors = np.empty(final_rendering_cube.shape, dtype=object)
        colors[building_zone_cube] = '#7A88CCC0'
        colors[agent_position_cube] = '#FFD65DC0'
        #print(colors)

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.voxels(final_rendering_cube, facecolors=colors, edgecolor='k')

        target_cube = self.target == 1
        # set the colors of each object
        colors = np.empty(target_cube.shape, dtype=object)
        colors[target_cube] = '#7A88CCC0'
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.voxels(target_cube, facecolors=colors, edgecolor='k')
        
        plt.show()
    
    """def define_new_target(self):
        empty_zone = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.voxels(empty_zone, facecolors='#7A88CCC0', edgecolor='k')
        
        def on_click(event):
            pressed = ax.button_pressed
            ax.button_pressed = -1 # some value that doesn't make sense.
            coords = ax.format_coord(event.xdata, event.ydata) # coordinates string in the form x=value, y=value, z= value
            ax.button_pressed = pressed
            print(coords)
            print(float("-10"))
            
            x, y, z = coords.split(", ")
            
            if x[2] not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                x = x.split("=")[1]
                x = "-" + x[1:]
                x = ceil(float(x))
            else:
                x = ceil(float(x.split("=")[1]))
            if y[2] not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                y = y.split("=")[1]
                y = "-" + y[1:]
                y = ceil(float(y))
            else:
                y = ceil(float(y.split("=")[1]))
            if z[2] not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                z = z.split("=")[1]
                z = "-" + z[1:]
                z = ceil(float(z))
            else:
                z = ceil(float(z.split("=")[1]))
           
            
            empty_zone[x, y, z] = 1
            target_cube = empty_zone == 1
            # set the colors of each object
            colors = np.empty(target_cube.shape, dtype=object)
            colors[target_cube] = '#7A88CCC0'
            ax.voxels(target_cube, facecolors=colors, edgecolor='k')
            plt.show()
            return coords        

        print("Click on the target position")
        cid = fig.canvas.mpl_connect('button_release_event', on_click)
        plt.show()"""
        
    def close(self):
        pass
    
    def get_sequence(self):
        return self.record_sequence
    
if __name__ == "__main__":
    # List of actions
    # 0: forward, 1: backward
    # 2: left, 3: right
    # 4: up, 5: down
    # 6: place block
    ##env.render()
    env = GridWorldEnv(4)
    #env.step(0)
    #env.step(6)
    #env.step(4)
    #env.step(6)
    #env.step(3)
    #env.step(6)
    #env.step(6)
    #env.step(4)
    #env.step(6)

    #env.step(3)
    #env.step(3)
    #env.step(3)
    #env.step(6)
    #env.step(2)
    #env.step(5)
    #env.step(3)
    #env.step(3)
    #env.step(3)
    #env.step(6)
    #env.step(4)
    #env.step(6)
    #env.step(2)
    #env.step(2)

    #env.step(4)
    #env.step(4)
    #env.step(4)
    #env.step(4)
    #env.step(6)
    #env.step(4)
    #env.step(6)
    #env.step(5)
    #env.step(5)
    #env.step(5)
    #env.step(3)


    #Test moving up
  
    env.render()

    #env.define_new_target()
    
