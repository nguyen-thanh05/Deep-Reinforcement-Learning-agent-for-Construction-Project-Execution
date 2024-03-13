import time
from tkinter import Grid
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from target_loader import TargetLoader
import random
MAX_TIMESTEP = 850


class GridWorldEnv(gym.Env):
    neighbors = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
    COL_BLOCK = 1
    BEAM_BLOCK = 2
    EMPTY = 0
    def __init__(self, dimension_size, path: str):
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, dimension_size, dimension_size, dimension_size), dtype=np.uint8)
        self.dimension_size = dimension_size
        self.timestep_elapsed = 0
        self.record_sequence = []

        self.agent_pos = [0, 0, 0]
        self.obs = np.zeros((3, self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)

        self.building_zone = self.obs[0]
        self.agent_pos_grid = self.obs[1]
        self.target = self.obs[2]
        self.all_targets = None

        self._initialized = False
        self.loader = TargetLoader(path)
        self.reset()

    def reset(self, seed=None, options=None):
        self.building_zone.fill(0)

        # self.agent_pos_grid.fill(0)
        self.agent_pos_grid[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] = 0
        self.agent_pos_grid[0, 0, 0] = 1
        self.agent_pos = [0, 0, 0]

        self.record_sequence = []
        # List of actions
        # 0: forward, 1: backward
        # 2: left, 3: right
        # 4: up, 5: down
        # 6: pick
        self.action_space = spaces.Discrete(8)
        self._init_obs()

        self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)

        self.timestep_elapsed = 0

        np.copyto(self.obs[2], random.choice(self.all_targets))
        return self.get_obs(), {}

    def get_obs(self):
        # clear agent_pos_grid
        # agent_pos = self.obs[1]

        self.agent_pos_grid.fill(0)
        self.agent_pos_grid[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] = 1

        return self.obs

    def _get_info(self):
        pass

    def _init_obs(self):
        if self._initialized:
            return
        self.all_targets = self.loader.load_all()

        assert len(self.all_targets) > 0, "No target found\n"
        for i in range(len(self.all_targets)):
            assert self.all_targets[i].shape[0] == self.dimension_size, \
                (f"Dimension mismatch: Target: {self.all_targets[i].shape}, "
                 f"Environment: {self.dimension_size}\n"
                 "TODO: more flexibility")
        self._initialized = True

    def step(self, action):
        self.timestep_elapsed += 1

        # List of actions
        # 0: forward, 1: backward
        # 2: left, 3: right
        # 4: up, 5: down
        # 6: place beam
        # 7: place col block
        if action == 1:
            # Y - 1
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 0:
            # Y + 1
            self.agent_pos[1] = min(self.dimension_size - 1, self.agent_pos[1] + 1)

        elif action == 2:
            # X - 1
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)

        elif action == 3:
            # X + 1
            self.agent_pos[0] = min(self.dimension_size - 1, self.agent_pos[0] + 1)

        elif action == 4:
            # Z + 1
            self.agent_pos[2] = min(self.dimension_size - 1, self.agent_pos[2] + 1)

        elif action == 5:
            # Z - 1
            self.agent_pos[2] = max(self.agent_pos[2] - 1, 0)

        elif action == 6:  # Place a beam block
            # First check for finished columns:
            if not self._check_columns_finish():
                duplicate_block = True
                supporting_neighbour = False
            else:
                # Find all 6 neighbouring directions
                neighbour_direction = [
                    [self.agent_pos[0] + delta_x, self.agent_pos[1] + delta_y, self.agent_pos[2] + delta_z]
                    for delta_x, delta_y, delta_z in GridWorldEnv.neighbors
                ]

                # Find if there is any supporting neighbour, or on the ground
                if (supporting_neighbour := self.agent_pos[2] == 0) is False:
                    for neighbour in neighbour_direction:
                        if neighbour[0] < 0 or neighbour[0] >= self.dimension_size \
                                or neighbour[1] < 0 or neighbour[1] >= self.dimension_size \
                                or neighbour[2] < 0 or neighbour[2] >= self.dimension_size:
                            continue

                        if self.building_zone[neighbour[0], neighbour[1], neighbour[2]] != GridWorldEnv.EMPTY:
                            supporting_neighbour = True
                            break

                # If the space is already occupied
                duplicate_block = self.building_zone[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] != GridWorldEnv.EMPTY
                if supporting_neighbour and not duplicate_block:
                    self.building_zone[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] = GridWorldEnv.BEAM_BLOCK
        
        elif action == 7: # Place a column block
            
            if self.building_zone[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] != GridWorldEnv.EMPTY:
                duplicate_block = True
                supporting_neighbour = False
                
            else:
                duplicate_block = False            

                if self.agent_pos[2] == 0: # If the agent is on the ground
                    self.building_zone[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] = GridWorldEnv.COL_BLOCK
                    supporting_neighbour = True
                else:
                    if self.building_zone[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2] - 1] == GridWorldEnv.COL_BLOCK:
                        supporting_neighbour = True
                        self.building_zone[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] = GridWorldEnv.COL_BLOCK
                    else:
                        supporting_neighbour = False
        

        # return observation, reward, terminated, truncated, info
        if action < 6:
            return self.get_obs(), -1, False, self.timestep_elapsed > MAX_TIMESTEP, {}
        # elif place_cmd:
        else:
            if supporting_neighbour and not duplicate_block:
                """
                --------------------------------------------------------                      
                | Built /Target |     0       |    1          |     2
                --------------------------------------------------------
                |      0        |    0        |      -1       |   -2
                |      1        |     1       |        0      |   -1
                |      2        |      2      |         1     |    0
                """
                check_done = np.isin(self.building_zone[self.target != 0], 0)
                if not np.any(check_done):
                    return self.get_obs(), 10, True, False, {}
                else:
                    if self.building_zone[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] == self.target[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]]:
                        return self.get_obs(), -0.25, False, self.timestep_elapsed > MAX_TIMESTEP, {}
                    else:
                        return self.get_obs(), -2, False, self.timestep_elapsed > MAX_TIMESTEP, {}
            elif duplicate_block or not supporting_neighbour:
                return self.get_obs(), -5, False, self.timestep_elapsed > MAX_TIMESTEP, {}

    def render(self):
        agent_pos_grid = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        agent_pos_grid[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] = 1

        # prepare some coordinates
        col_cube = self.building_zone == GridWorldEnv.COL_BLOCK
        beam_cube = self.building_zone == GridWorldEnv.BEAM_BLOCK
        agent_position_cube = agent_pos_grid == 1

        fig = plt.figure()

        building_zone_render = col_cube | agent_position_cube | beam_cube
        # set the colors of each object
        colors = np.empty(building_zone_render.shape, dtype=object)
        colors[col_cube] = '#7A88CCC0'
        colors[agent_position_cube] = '#FFD65DC0'
        colors[beam_cube] = '#FF5733C0'
        # print(colors)

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.voxels(building_zone_render, facecolors=colors, edgecolor='k')

        col_cube = self.target == GridWorldEnv.COL_BLOCK
        beam_cube = self.target == GridWorldEnv.BEAM_BLOCK
        target_render = col_cube | beam_cube
        # set the colors of each object
        colors = np.empty(target_render.shape, dtype=object)
        colors[col_cube] = '#7A88CCC0'
        colors[beam_cube] = '#FF5733C0'
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.voxels(target_render, facecolors=colors, edgecolor='k')

        plt.show()

    
    def _check_columns_finish(self):
        difference = self.building_zone - self.target
        difference = np.isin(difference, -GridWorldEnv.COL_BLOCK)
        if np.any(difference):
            return False
        return True
    
    def close(self):
        pass

    def get_sequence(self):
        return self.record_sequence


if __name__ == "__main__":
    # List of actions
    # 0: forward, 1: backward
    # 2: left, 3: right
    # 4: up, 5: down
    # 6: place beam
    # 7: place col block
    
    env = GridWorldEnv(4, "targets")
    env.building_zone[:, :, :3] = GridWorldEnv.COL_BLOCK
    env.step(4)
    env.step(4)
    env.step(4)
    env.step(4)
    
    env.step(6)
    env.step(7)
    env.step(5)
    env.step(6)
    env.step(4)
    env.render()
    print(env.get_obs())
