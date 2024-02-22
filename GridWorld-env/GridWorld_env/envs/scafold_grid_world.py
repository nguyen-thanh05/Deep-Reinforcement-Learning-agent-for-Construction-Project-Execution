
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
# import enum
from enum import Enum

class Action:
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    UP = 4
    DOWN = 5
    PLACE_SCAFOLD = 6
    REMOVE_SCAFOLD = 7
    PLACE_FORWARD = 8
    PLACE_BACKWARD = 9
    PLACE_LEFT = 10
    PLACE_RIGHT = 11


class ScaffoldGridWorldEnv(gym.Env): 
    """
        3 TUPLE element state
        1. agent position
        3. building zone
        4. target zone

        block description:
        0: empty
        -1: block
        -2: scaffold
        1: agents/ or should 

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
        self.action_enum = Action
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

        self._placeBlockInBuildingZone()  # TODO: REMOVE
        self.AgentsPos = np.zeros((self.num_agents, 3), dtype=int)

        
        random_start_pos = np.zeros(3, dtype=int)
        for i in range(self.num_agents):
            random_start_pos[0] = np.random.randint(0, self.dimension_size)
            random_start_pos[1] = np.random.randint(0, self.dimension_size)
            random_start_pos[2] = 0
            self.AgentsPos[i] = random_start_pos
            #self.building_zone[random_start_pos[0], random_start_pos[1], random_start_pos[2]] = 1  # encode agents position on the building zone

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

    def _placeBlockInBuildingZone(self):
        # place some block in building zone for testing
        self.building_zone[self.dimension_size // 2, self.dimension_size // 2, 0] = -1
        return

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

    def _tryMove(self, action, agent_id):
        if (action in [0, 1, 2, 3, 4, 5]):
            new_pos = self.AgentsPos[agent_id].copy()
            if (action == 0):
                new_pos[0] += 1
            elif (action == 1):
                new_pos[0] -= 1
            elif (action == 2):
                new_pos[1] -= 1
            elif (action == 3):
                new_pos[1] += 1
            elif (action == 4):
                new_pos[2] += 1
            elif (action == 5):
                new_pos[2] -= 1
            return new_pos

        return None

    def _isInScaffoldingDomain(self, pos):
        if (self.building_zone[pos[0], pos[1], pos[2]] == -2):
            return True
        return False 
    def _isInBlock(self, pos):
        if (self.building_zone[pos[0], pos[1], pos[2]] == -1):
            return True
        return False
    # position is not in any block or scaffolding
    def _isInNothing(self, pos):
        if (self.building_zone[pos[0], pos[1], pos[2]] == 0):
            return True
        return False
    # there is an agent in the position 
    def _thereIsAgent(self, pos):
        if (self.building_zone[pos[0], pos[1], pos[2]] == 1):
            return True
        return False

    """
    check if NWSE is move is valid
    prev_pos -> action -> new_pos 
    """    
    def _isValidMove(self, new_pos, action , prev_pos):
        if (new_pos[0] < 0 or new_pos[0] >= self.dimension_size or new_pos[1] < 0 or new_pos[1] >= self.dimension_size or new_pos[2] < 0 or new_pos[2] >= self.dimension_size):
            return False
        # case: it moved out of the block
        if (new_pos[2] > 0 and self.building_zone[new_pos[0], new_pos[1], new_pos[2] - 1] == 0):  # case: there is no block below and we are above the ground
            return False
        if (self.building_zone[new_pos[0], new_pos[1], new_pos[2]] == -1):  # there is a block here
            return False
        # check if  
        if (action == self.action_enum.UP):
            # if prev_pos is in scaffolding domain and new_pos is not in block(so new pos is air or another scaffolding)
            if (self._isInScaffoldingDomain(prev_pos) and not self._isInBlock(new_pos)):  # if agent was in the scaffolding domain
                return True
            return False
        if (action == self.action_enum.DOWN):
            # case: scafold before and after
            if (self._isInScaffoldingDomain(prev_pos) and self._isInScaffoldingDomain(new_pos)):
                return True
            # case: we on top of scafold and there is scafold below
            if (not self._isInScaffoldingDomain(prev_pos) and self._isInScaffoldingDomain(new_pos)):
                return True
            return False
        # case: handle climbing down and up a block
        return True
    def _isValidPlace(self, action, current_pos, agent_id):
        if (action == self.action_enum.PLACE_SCAFOLD):
            if (not self._isInScaffoldingDomain(current_pos) and not self._isInBlock(current_pos)):  # case: there is not
                return True 

        # case: place block
            
        
        return False
    def step(self, action_tuple):
        if (len(action_tuple) != 2):
            raise ValueError("action_tuple should be a tuple of 2 elements")

        action = action_tuple[0]
        agent_id = action_tuple[1]
        self.mutex.acquire()  # get lock to enter critical section
        self.timestep_elapsed += 1
        """
        ACTION:
        0: forward, 1: backward, 2: left, 3: right                         [move]
        4: up, 5: down                                                     [move but only in the scaffolding domain]
        6: place scaffold at current position                              [place block]
        7: remove scaffold at current position                             [remove block]
        8-11: place block at the 4 adjacent position of the agent          [place block]
        """
        current_pos = self.AgentsPos[agent_id]
        new_pos = self._tryMove(action, agent_id)

        if (action in [0, 1, 2, 3, 4, 5]):  # move action
            if (self._isValidMove(new_pos, action, current_pos)):
                #self.building_zone[self.AgentsPos[agent_id][0], self.AgentsPos[agent_id][1], self.AgentsPos[agent_id][2]] = 0
                self.AgentsPos[agent_id][0] = new_pos[0]
                self.AgentsPos[agent_id][1] = new_pos[1]
                self.AgentsPos[agent_id][2] = new_pos[2]
                #self.building_zone[self.AgentsPos[agent_id][0], self.AgentsPos[agent_id][1], self.AgentsPos[agent_id][2]] = 1

                obs = self.get_obs(agent_id)
                self.mutex.release()
                if self.timestep_elapsed > MAX_TIMESTEP:
                    return self.get_obs(agent_id), -0.5, False, True, {}
                else:
                    return self.get_obs(agent_id), -0.5, False, False, {}

            else:  # case: invalid move, so agent just stay here
                obs = self.get_obs(agent_id)
                self.mutex.release()
                if self.timestep_elapsed > MAX_TIMESTEP:
                    return obs, -1, False, True, {}
                else:
                    return obs, -1, False, False, {}
        elif (action == self.action_enum.PLACE_SCAFOLD):
            # agent can only place scaffold if there is nothing in current position
            if (self._isValidPlace(action, current_pos, agent_id)):
                self.building_zone[current_pos[0], current_pos[1], current_pos[2]] = -2  # place scaffold block
                obs = self.get_obs(agent_id)
                self.mutex.release()
                if self.timestep_elapsed > MAX_TIMESTEP:
                    return obs, -0.5, False, True, {}
                else:
                    return obs, -0.5, False, False, {}
            else:  # case: invalid place, so agent just stay here
                obs = self.get_obs(agent_id)
                self.mutex.release()
                if self.timestep_elapsed > MAX_TIMESTEP:
                    return obs, -1, False, True, {}
                else:
                    return obs, -1, False, False, {}
            pass
        elif (action == self.action_enum.REMOVE_SCAFOLD):
            # agent can only remove scaffold if there is a scaffold in current position and there is no scaffold above or agent above
            if (self._isInScaffoldingDomain(current_pos)
             and not self._isInScaffoldingDomain([current_pos[0], current_pos[1], current_pos[2] + 1])
             and not self._thereIsAgent([current_pos[0], current_pos[1], current_pos[2] + 1])):

                self.building_zone[current_pos[0], current_pos[1], current_pos[2]] = 0
                obs = self.get_obs(agent_id)
                self.mutex.release()
                if self.timestep_elapsed > MAX_TIMESTEP:
                    return obs, -1, False, True, {}
                else:
                    return obs, -1, False, False, {}
            else:  # case: invalid remove, so agent just stay here
                obs = self.get_obs(agent_id)
                self.mutex.release()
                if self.timestep_elapsed > MAX_TIMESTEP:
                    return obs, -1, False, True, {}
                else:
                    return obs, -1, False, False, {}
        elif (action in [8, 9, 10, 11]):  # place command
            pass



 
    

    def render(self):
        # acumulate all agents position
        agent_pos_grid = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        for i in range(self.num_agents):
            agent_pos_grid[self.AgentsPos[i][0], self.AgentsPos[i][1], self.AgentsPos[i][2]] = 1

        # prepare some coordinates
        building_zone_cube = self.building_zone == -1
        agent_position_cube = agent_pos_grid == 1
        scaffold_cube = self.building_zone == -2

        fig = plt.figure()

        final_rendering_cube = building_zone_cube | agent_position_cube | scaffold_cube
        # set the colors of each object
        colors = np.empty(final_rendering_cube.shape, dtype=object)
        colors[building_zone_cube] = '#7A88CCC0'
        colors[agent_position_cube] = '#FFD65DC0'
        colors[scaffold_cube] = 'pink'
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
    


def testMove(env, agent_id):
    # move forward
    for i in range(5):
        action = i
        env.step((action, agent_id))
        env.render()
def testScafold(env, agent_id):
    env.render()
    # move forward
    env.step((0, agent_id))
    env.render()
    # place scaffold
    env.step((6, agent_id))
    env.render()
    # move up
    env.step((4, agent_id))
    env.render()
    # move down 
    env.step((5, agent_id))
    env.render()
    # remove scafold    
    env.step((7, agent_id))
    env.render()

    return
if __name__ == "__main__":
    # List of actions
    # 0: forward, 1: backward
    # 2: left, 3: right
    # 4: up, 5: down
    # 6: place block
    env = ScaffoldGridWorldEnv(4, 1)

    # test move
    #testMove(env, 0)
    testScafold(env, 0)
    #env.step(0)
    #env.step(6)
    #env.step(4)
    #env.step(6)
    #env.step(3)
    #env.step(6)
    ##env.render()
    #print(env.building_zone)

    
    
