
from math import ceil
from turtle import onclick
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import torch
MAX_TIMESTEP = 500


#import Lock
from threading import Lock
class ScaffoldGridWorldEnv(gym.Env): 
    """
        3 TUPLE element state
        1. agent position
        3. building zone
        4. target zone

        block description:
        0: empty
        -1: block
        1: agents/ or should 
        -2: scaffold

        ACTION:
        0: forward, 1: backward, 2: left, 3: right                         [move]
        4: up, 5: down                                                     [move but only in the scaffolding domain]
        6: place scaffold at current position                              [place block]
        7: remove scaffold at current position                             [remove block]
        8-11: place block at the 4 adjacent position of the agent          [place block]


        Rule:
        1. agent can move in 4 direction (N, S, E, W), and 2 direction in the z axis (up, down) agent in scaffolding domain
        2. agent cannot move through block
        3. if agent is moving to a coordinate with a block adjacent to it, it will climb on top of it
        4. agent can climb down a block if agent is 1 block above the ground, otherwise agent will die

    """
    def __init__(self, dimension_size, num_agents=1):
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.dimension_size = dimension_size
        self.timestep_elapsed = 0
        self.mutex = Lock()
        self.num_agents = num_agents

        self.reset()

    
    # in multiagent, make sure only one agent is allowed to call this function(meta agent for example)
    def reset(self, seed=None, options=None):
        self.mutex.acquire()  # get lock to enter critical section
        self.building_zone = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)

        self.AgentsPos = np.zeros((self.num_agents, 3), dtype=int)

        
        random_start_pos = np.zeros(3, dtype=int)
        for i in range(self.num_agents):
            random_start_pos[0] = np.random.randint(0, self.dimension_size)
            random_start_pos[1] = np.random.randint(0, self.dimension_size)
            random_start_pos[2] = np.random.randint(0, self.dimension_size)
            self.AgentsPos[i] = random_start_pos
            self.building_zone[random_start_pos[0], random_start_pos[1], random_start_pos[2]] = 1  # encode agents position on the building zone

        #self.agent_pos = [random_start_pos[0], random_start_pos[1], random_start_pos[2]]
        # List of actions
        # 0: forward, 1: backward
        # 2: left, 3: right
        # 4: up, 5: down
        # 6: pick
        self.action_space = spaces.Discrete(12)   
        self._init_target()
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)

        self.timestep_elapsed = 0


        obs = self.get_obs(0)
        self.mutex.release()
        return obs, {}

    # return (3 x N x N x N) tensor for now, 
    def get_obs(self, agent_id):
        # clear agent_pos_grid
        agent_pos_grid = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        agent_pos_grid[self.AgentsPos[agent_id][0], self.AgentsPos[agent_id][1], self.AgentsPos[agent_id][2]] = 1


        other_agents_pos_grid = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        for i in range(self.num_agents):
            if i != agent_id:
                other_agents_pos_grid[self.AgentsPos[i][0], self.AgentsPos[i][1], self.AgentsPos[i][2]] = 1

        #TODO: concat other_agents_pos_grid when doing multiagent
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
            self.target[p[0], p[1], p[2]] = -1  # -1 is block

    def step(self, action_tuple):
        if (len(action_tuple) != 2):
            raise ValueError("action_tuple should be a tuple of 2 elements")

        action = action_tuple[0]
        agent_id = action_tuple[1]
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
                # If the space is already occupied
                if self.building_zone[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] == 1:
                    supporting_neighbour = False
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
                return self.get_obs(), -0.5, False, True, {}
            else:
                return self.get_obs(), -0.5, False, False, {}
        elif place_cmd:
            if supporting_neighbour:
                difference = self.target - self.building_zone
                difference = np.isin(difference, 1)
                if np.any(difference) == False:
                    return self.get_obs(), 0, True, False, {}
                else:
                    if self.building_zone[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] == self.target[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]]:
                        if self.timestep_elapsed > MAX_TIMESTEP:
                            return self.get_obs(), 0.5, False, True, {}
                        else:
                            return self.get_obs(), 0.5, False, False, {}
                    else:
                        if self.timestep_elapsed > MAX_TIMESTEP:
                            return self.get_obs(), -1, False, True, {}
                        else:
                            return self.get_obs(), -1, False, False, {}
            else:
                if self.timestep_elapsed > MAX_TIMESTEP:
                    return self.get_obs(), -2.5, False, True, {}        
                else:
                    return self.get_obs(), -2.5, False, False, {}
 
    

    def render(self):
        # acumulate all agents position
        agent_pos_grid = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        for i in range(self.num_agents):
            agent_pos_grid[self.AgentsPos[i][0], self.AgentsPos[i][1], self.AgentsPos[i][2]] = 1

        # prepare some coordinates
        building_zone_cube = self.building_zone == -1
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
    
    def close(self):
        pass
    
if __name__ == "__main__":
    # List of actions
    # 0: forward, 1: backward
    # 2: left, 3: right
    # 4: up, 5: down
    # 6: place block
    env = ScaffoldGridWorldEnv(4, 2)
    env.render()
    #env.step(0)
    #env.step(6)
    #env.step(4)
    #env.step(6)
    #env.step(3)
    #env.step(6)
    ##env.render()
    #print(env.building_zone)

    
    
