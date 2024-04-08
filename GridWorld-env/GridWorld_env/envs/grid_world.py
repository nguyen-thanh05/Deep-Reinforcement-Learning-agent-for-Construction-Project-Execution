import gymnasium as gym
import random
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
MAX_TIMESTEP = 3000
from target_loader import TargetLoader
import io

class Action:
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    UP = 4
    DOWN = 5
    PLACE_SCAFOLD = 6
    REMOVE_SCAFOLD = 7
    PLACE_BEAM = 8
    PLACE_COLUMN = 9


class GridWorldEnv(gym.Env): 
    neighbors = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
    SCAFFOLD = -1
    EMPTY = 0

    COL_BLOCK = 0.5
    BEAM_BLOCK = 1
    
    def __init__(self, dimension_size, batch_size, path: str):
        self.action_enum = Action
        #self.reward = Reward()
        self.batch_size = batch_size
        self.dimension_size = dimension_size
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.batch_size, 3, self.dimension_size, self.dimension_size, self.dimension_size), dtype=float)

        self.timestep_elapsed = 0
        self.finished_structure = False
        self.finished_structure_with_scafold = np.zeros((self.batch_size,), dtype=bool)
        self.finish_columns_one_time_reward = np.zeros((self.batch_size,), dtype=bool)
        
        self.record_sequence = []
        
        self.obs = np.zeros((self.batch_size, 3, self.dimension_size, self.dimension_size, self.dimension_size), dtype=float)
        
        self.all_targets = None
        self._initialized = False
        self.done_array = np.zeros((self.batch_size,), dtype=bool)

        self.building_zone = self.obs[:, 0]
        self.agent_pos_grid = self.obs[:, 1]
        
        self.target = self.obs[:, 2]
        
        self.loader = TargetLoader(path)
        
        self.reset()
        
            
    def reset(self, seed=None, options=None):
        self.building_zone.fill(0)
        self.finished_structure = False
        self.finish_columns_one_time_reward = np.zeros((self.batch_size,), dtype=bool)
        self.finished_structure_with_scafold = np.zeros((self.batch_size,), dtype=bool)

        self.done_array = np.zeros((self.batch_size,), dtype=bool)
        
        self.agent_pos = np.zeros((self.batch_size, 3), dtype=int)

        for i in range(self.batch_size):
            random_start_pos = np.zeros(3, dtype=int)
            random_start_pos[0] = np.random.randint(0, self.dimension_size)
            random_start_pos[1] = np.random.randint(0, self.dimension_size)
            random_start_pos[2] = np.random.randint(0, self.dimension_size)
            self.agent_pos[i] = random_start_pos
            
        
        self.action_space = spaces.Discrete(10)   
        #self._init_target()
        
        self.record_sequence = []
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.batch_size, 3, self.dimension_size, self.dimension_size, self.dimension_size), dtype=float)

        self.timestep_elapsed = 0
        self._init_obs()

        for i in range(self.batch_size):
            individual_target = random.choice(self.all_targets)
            np.copyto(self.obs[i, 2], individual_target)

        for i in range(self.batch_size):
            for j in range(self.dimension_size):
                for k in range(self.dimension_size):
                    for l in range(self.dimension_size):
                        if self.obs[i][2][j][k][l] == 1:
                            self.obs[i][2][j][k][l] = GridWorldEnv.COL_BLOCK
                        elif self.obs[i][2][j][k][l] == 2:
                            self.obs[i][2][j][k][l] = GridWorldEnv.BEAM_BLOCK
        

        obs = self.get_obs()
        
        return obs, {}

    # return (3 x N x N x N) tensor for now, 
    def get_obs(self):
        # clear agent_pos_grid
        self.agent_pos_grid.fill(0)
        
        for i in range(self.batch_size):
            self.agent_pos_grid[i][self.agent_pos[i][0], self.agent_pos[i][1], self.agent_pos[i][2]] = 1
        return self.obs
        
    def _get_info(self):
        pass

    def _init_obs(self):
        if self._initialized:
            return
        self.all_targets = self.loader.load_all()
        # convert all targets to float
    

        assert len(self.all_targets) > 0, "No target found\n"
        for i in range(len(self.all_targets)):
            assert self.all_targets[i].shape[0] == self.dimension_size, \
                (f"Dimension mismatch: Target: {self.all_targets[i].shape}, "
                 f"Environment: {self.dimension_size}\n"
                 "TODO: more flexibility")
        self._initialized = True


    def _change_agent_position(self, action, agent_id):
        #if (action in [0, 1, 2, 3, 4, 5]):
        new_pos = self.agent_pos[agent_id].copy()
        if (action == 0):  # move forward
            new_pos[1] += 1
        elif (action == 1):  # move backward
            new_pos[1] -= 1
        elif (action == 2):  # move left
            new_pos[0] -= 1
        elif (action == 3):  # move right
            new_pos[0] += 1
        elif (action == 4):  # move up
            new_pos[2] += 1
        elif (action == 5):  # move down
            new_pos[2] -= 1
        return new_pos


    def _is_scaffold(self, pos, env_id):
        if (self.building_zone[env_id, pos[0], pos[1], pos[2]] == GridWorldEnv.SCAFFOLD):
            return True
        return False 
    
    
    def _isInBlock(self, pos):
        if (self.building_zone[pos[0], pos[1], pos[2]] == GridWorldEnv.COL_BLOCK or self.building_zone[pos[0], pos[1], pos[2]] == GridWorldEnv.BEAM_BLOCK):
            return True
        return False
    
    
    def _is_col(self, pos, env_id):
        if self.building_zone[env_id, pos[0], pos[1], pos[2]] == GridWorldEnv.COL_BLOCK:
            return True
        return False


    def _beamExist(self, pos):
        if (self.building_zone[pos[0], pos[1], pos[2]] == GridWorldEnv.BEAM_BLOCK):
            return True
        return False


    def _is_empty(self, pos, env_id):
        if (self.building_zone[env_id, pos[0], pos[1], pos[2]] == 0):
            return True
        return False
   

    def _is_pos_valid(self, new_pos):
        if (new_pos[0] < 0 or new_pos[0] >= self.dimension_size or new_pos[1] < 0 or new_pos[1] >= self.dimension_size or new_pos[2] < 0 or new_pos[2] >= self.dimension_size):
            return False
        
        return True
    

    def _is_scaffold_valid(self, current_pos, env_id):
        neighbour_deltas = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1]]
                              # LEFT       RIGHT       BEHIND      FRONT     DOWN             
        neighbour_direction = [
                [current_pos[0] + delta_x, current_pos[1] + delta_y, current_pos[2] + delta_z]
                for delta_x, delta_y, delta_z in neighbour_deltas
            ]

        # Find if there is any supporting neighbour, or on the ground
        supporting_neighbour = current_pos[2] == 0
        if supporting_neighbour == False:
            for neighbour in neighbour_direction:
                if neighbour[0] < 0 or neighbour[0] >= self.dimension_size \
                        or neighbour[1] < 0 or neighbour[1] >= self.dimension_size \
                        or neighbour[2] < 0 or neighbour[2] >= self.dimension_size:
                    continue

                if self.building_zone[env_id, neighbour[0], neighbour[1], neighbour[2]] == GridWorldEnv.SCAFFOLD:
                    supporting_neighbour = True
                    break

        # If the space is already occupied
        duplicate_block = self.building_zone[env_id, current_pos[0], current_pos[1], current_pos[2]] != GridWorldEnv.EMPTY
        if supporting_neighbour and not duplicate_block:
            return True
        return False
    

    def _check_support(self, currentPos, env_id, is_beam=False):   
        support = True
        scalffold_direction = [  
            [currentPos[0] + delta_x, currentPos[1] + delta_y, currentPos[2] + delta_z]
            for delta_x, delta_y, delta_z in [[-1, 0, -1], [1, 0, -1], [0, -1, -1], [0, 1, -1], [0, 0, -1]]
        ]                                       # LEFT       RIGHT       BEHIND      FRONT

        adjacent_direction = [
            [currentPos[0] + delta_x, currentPos[1] + delta_y, currentPos[2] + delta_z]
            for delta_x, delta_y, delta_z in [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]]
        ]                                       # LEFT       RIGHT       BEHIND      FRONT
        
        # Remove invalid directions
        for direction in scalffold_direction:
            if direction[0] < 0 or direction[0] >= self.dimension_size \
                    or direction[1] < 0 or direction[1] >= self.dimension_size \
                    or direction[2] < 0 or direction[2] >= self.dimension_size:
                scalffold_direction.remove(direction)
        for direction in adjacent_direction:
            if direction[0] < 0 or direction[0] >= self.dimension_size \
                    or direction[1] < 0 or direction[1] >= self.dimension_size \
                    or direction[2] < 0 or direction[2] >= self.dimension_size:
                adjacent_direction.remove(direction)
        
        for scaffold_dir in scalffold_direction:
            if self.building_zone[env_id, scaffold_dir[0], scaffold_dir[1], scaffold_dir[2]] == GridWorldEnv.EMPTY:
                support = False
                break
        if is_beam:
            if support:
                if self.building_zone[env_id, currentPos[0], currentPos[1], currentPos[2] - 1] == GridWorldEnv.COL_BLOCK:
                    support = True
                else:
                    support = False
                    for adjacent_dir in adjacent_direction:
                        if self.building_zone[env_id, adjacent_dir[0], adjacent_dir[1], adjacent_dir[2]] == GridWorldEnv.COL_BLOCK or \
                            self.building_zone[env_id, adjacent_dir[0], adjacent_dir[1], adjacent_dir[2]] == GridWorldEnv.BEAM_BLOCK:
                            support = True
                            break
        return support
        
        
    # done and there is no more scaffold
    def _is_building_done_no_scaffold(self, env_id):
        # check if col is finished
        col_done = self._check_finish_columns(env_id)
        beam_done = self._check_finish_beams(env_id)
    
        scaffold_left = np.any(np.isin(self.building_zone[self.target == 0], GridWorldEnv.SCAFFOLD))
        if col_done and beam_done and not scaffold_left:
            return True
        return False
    
    
    # done structure but there is scafold left 
    def _is_building_done_with_scaffold(self, env_id):
        # check if col is finished
        col_done = self._check_finish_columns(env_id)
        beam_done = self._check_finish_beams(env_id)
    
        if col_done and beam_done: #and not scaffold_left:
            return True
        return False
        
    
    def _check_finish_beams(self, env_id):
        check = np.isin(self.building_zone[env_id][self.target[env_id] == GridWorldEnv.BEAM_BLOCK], GridWorldEnv.BEAM_BLOCK)
        if np.all(check):
            return True
        return False
    
    
    def _check_finish_columns(self, env_id):
        check = np.isin(self.building_zone[env_id][self.target[env_id] == GridWorldEnv.COL_BLOCK], GridWorldEnv.COL_BLOCK)
        if np.all(check):
            return True
        return False
    

    def _step(self, action, env_id):
        self.timestep_elapsed += 1
        
        current_pos = self.agent_pos[env_id]

        if (action in [self.action_enum.FORWARD, 
                       self.action_enum.BACKWARD,
                       self.action_enum.RIGHT,
                       self.action_enum.LEFT,
                       self.action_enum.UP,
                       self.action_enum.DOWN]):  # move action
            reward_signal = -0.28
            terminated = False
            truncated = False
            is_valid = False
            new_pos = self._change_agent_position(action, env_id)
            if (self._is_pos_valid(new_pos)):
                
                self.agent_pos[env_id][0] = new_pos[0]
                self.agent_pos[env_id][1] = new_pos[1]
                self.agent_pos[env_id][2] = new_pos[2]
                
            
            if self.timestep_elapsed > MAX_TIMESTEP:
                truncated = True    
            return reward_signal, self.done_array[env_id], truncated, {}

        elif (action == self.action_enum.PLACE_SCAFOLD):
            reward_signal = -0.3
            terminated = False
            truncated = False
            is_valid = False
            
            if (self._is_scaffold_valid(current_pos, env_id)):
                self.building_zone[env_id, current_pos[0], current_pos[1], current_pos[2]] = GridWorldEnv.SCAFFOLD 
                is_valid = True

            if not is_valid: reward_signal = -1
            if self.timestep_elapsed > MAX_TIMESTEP:
                truncated = True

            return reward_signal, self.done_array[env_id], truncated, {}
            
        elif (action == self.action_enum.REMOVE_SCAFOLD):
            reward_signal = -0.3
            if self.finished_structure_with_scafold[env_id]:
                reward_signal = 0.2
            terminated = False
            truncated = False
            is_valid = False
            # agent can only remove scaffold if there is a scaffold in current position and there is no scaffold above or agent above

            
            if (self._is_scaffold(current_pos, env_id)):
                if self._is_pos_valid([current_pos[0], current_pos[1], current_pos[2] + 1]) and (self._is_empty([current_pos[0], current_pos[1], current_pos[2] + 1], env_id) or
                                                                                                 self.building_zone[env_id, current_pos[0], current_pos[1], current_pos[2] + 1] == GridWorldEnv.BEAM_BLOCK):
                
                    # case: remove scaffold is not on the top floor and there is no block above
                    self.building_zone[env_id, current_pos[0], current_pos[1], current_pos[2]] = GridWorldEnv.EMPTY
                    is_valid = True
                elif not self._is_pos_valid([current_pos[0], current_pos[1], current_pos[2] + 1]):
                    # case: remove scaffold is on the top floor
                    self.building_zone[env_id, current_pos[0], current_pos[1], current_pos[2]] = GridWorldEnv.EMPTY
                    is_valid = True
            else:  # case: invalid remove, so agent just stay here
                pass
                
            if not is_valid: reward_signal = -1
            if self.timestep_elapsed > MAX_TIMESTEP:
                truncated = True
                return reward_signal, self.done_array[env_id], truncated, {}
            else:
                if is_valid and self._is_building_done_no_scaffold(env_id) and not self.done_array[env_id]:
                    reward_signal = 1
                    self.done_array[env_id] = True
                return reward_signal, self.done_array[env_id], truncated, {}
        elif action == self.action_enum.PLACE_COLUMN:  # place command
            reward_signal = -0.35
            truncated = False
            is_valid = False
            
            # if there is already a block or a scaffold in the position
            if not self._is_empty(current_pos, env_id):
                is_valid = False
            
            # Check if there is proper support. Case 1, on the floor
            elif current_pos[2] == 0:
                is_valid = True
            # Case 2, on the scaffold and there is column block below
            elif self._check_support(current_pos, env_id, is_beam=False) and self._is_col((current_pos[0], current_pos[1], current_pos[2] - 1), env_id):
                is_valid = True
                
            if is_valid:
                self.building_zone[env_id, current_pos[0], current_pos[1], current_pos[2]] = GridWorldEnv.COL_BLOCK

                if  self.target[env_id, current_pos[0], current_pos[1], current_pos[2]] == self.building_zone[env_id, current_pos[0], current_pos[1], current_pos[2]]:
                    reward_signal = 0.9  # placing scafold column costs (-0.2 + -0.2) = -0.4. Placing column stack costs -1.2. (6*-0.2) = -1.2. so 0.9 + -1.2 > -0.4
                else:
                    reward_signal = -0.35
            else:
                reward_signal = -1
            
            # check if structure is complete
            if is_valid and self._check_finish_columns(env_id) and not self.finish_columns_one_time_reward[env_id]:  #  only do terminal check if we placed a block to save computation               
                reward_signal = 1
                self.finish_columns_one_time_reward[env_id] = True

            if self.timestep_elapsed > MAX_TIMESTEP:
                truncated = True
                return reward_signal, self.done_array[env_id], truncated, {}
            else:
                return reward_signal, self.done_array[env_id], truncated, {}
        elif action == self.action_enum.PLACE_BEAM:
            reward_signal = -0.35
            terminated = False
            truncated = False
            is_valid = False
            # case: havent fnished column block yet
            if not self._check_finish_columns(env_id):
                return -1, self.done_array[env_id], self.timestep_elapsed > MAX_TIMESTEP, {}
            else:
                # if there is already a block or a scaffold in the position
                if not self._is_empty(current_pos, env_id):
                    is_valid = False
                
                # Check if there is proper support. Case 1, on the floor
                elif current_pos[2] == 0:
                    is_valid = True
                # Case 2, on the scaffold
                elif self._check_support(current_pos, env_id, is_beam=True):
                    is_valid = True
                    
                if is_valid:
                    self.building_zone[env_id, current_pos[0], current_pos[1], current_pos[2]] = GridWorldEnv.BEAM_BLOCK
                    if  self.target[env_id, current_pos[0], current_pos[1], current_pos[2]] == self.building_zone[env_id, current_pos[0], current_pos[1], current_pos[2]]:
                        reward_signal = 0.9
                    else:
                        reward_signal = -0.35
                else:
                    reward_signal = -1

                if (is_valid and self._is_building_done_with_scaffold(env_id) and not self.finished_structure_with_scafold[env_id]):  #  only do terminal check if we placed a block to save computation
                    #terminated = True
                    reward_signal = 1
                    self.finished_structure_with_scafold[env_id] = True
                
                
                if self.timestep_elapsed > MAX_TIMESTEP:
                    truncated = True
                    return reward_signal, self.done_array[env_id], truncated, {}
                else:
                    return reward_signal, self.done_array[env_id], truncated, {}


    def step(self, action_array):
        reward_array = np.zeros((self.batch_size,), dtype=float)
        truncated_array = np.zeros((self.batch_size,), dtype=bool)
        
        for i, action_array in enumerate(action_array):
            reward, done, truncated, info = self._step(action_array, i)

            reward_array[i] = reward
            truncated_array[i] = truncated

        obs = self.get_obs()
        return obs, reward_array , self.done_array, truncated_array, {}

    def send_plot_to_tensorboard(self):
        fig  = plt.figure(figsize=(15, 10))
        agent_pos_grid = np.zeros((self.batch_size, self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        
        for i in range(5):
            agent_pos_grid[i, self.agent_pos[i][0], self.agent_pos[i][1], self.agent_pos[i][2]] = 1

            col_cube = self.building_zone[i] == GridWorldEnv.COL_BLOCK
            beam_cube = self.building_zone[i] == GridWorldEnv.BEAM_BLOCK
            scaffold_cube = self.building_zone[i] == GridWorldEnv.SCAFFOLD
            
            agent_position_cube = agent_pos_grid[i] == 1    
            building_zone_render = col_cube | agent_position_cube | beam_cube | scaffold_cube
            # set the colors of each object
            colors = np.empty(building_zone_render.shape, dtype=object)
            colors[col_cube] = '#7A88CCC0'
            colors[agent_position_cube] = '#FFD65DC0'
            colors[beam_cube] = '#FF5733C0'
            colors[scaffold_cube] = '#f3f6f4C0'
            
            ax = fig.add_subplot(1, 5, i + 1, projection='3d')
            ax.voxels(building_zone_render, facecolors=colors, edgecolor='k')
            ax.set_title(f"Agent {i+1}")
        return fig
        
    def render(self):
        fig  = plt.figure()
        agent_pos_grid = np.zeros((self.batch_size, self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        
        for i in range(self.batch_size):
            agent_pos_grid[i, self.agent_pos[i][0], self.agent_pos[i][1], self.agent_pos[i][2]] = 1

            col_cube = self.building_zone[i] == GridWorldEnv.COL_BLOCK
            beam_cube = self.building_zone[i] == GridWorldEnv.BEAM_BLOCK
            scaffold_cube = self.building_zone[i] == GridWorldEnv.SCAFFOLD
            
            agent_position_cube = agent_pos_grid[i] == 1    
            building_zone_render = col_cube | agent_position_cube | beam_cube | scaffold_cube
            # set the colors of each object
            colors = np.empty(building_zone_render.shape, dtype=object)
            colors[col_cube] = '#7A88CCC0'
            colors[agent_position_cube] = '#FFD65DC0'
            colors[beam_cube] = '#FF5733C0'
            colors[scaffold_cube] = '#f3f6f4C0'
            
            ax = fig.add_subplot(1, self.batch_size, i + 1, projection='3d')
            ax.voxels(building_zone_render, facecolors=colors, edgecolor='k')
            ax.set_title(f"Agent {i+1}")
        plt.show()


    def close(self):
        pass


def test_dim_4(env):
    env.agent_pos_grid.fill(0)
    env.agent_pos_grid[0, 0, 0] = 1 
    env.step((0, 0))
    env.step((3, 0))
    
    for i in range(3):
        env.step([9, 9])
        env.step([1, 1])
        env.step([6, 6])
        env.step([0, 0])
        env.step([2, 2])
        env.step([6, 6])
        env.step([3, 3])
        env.step([4, 4])
    
    env.step([2, 2])
    env.step([2, 2])
    env.step([2, 2])
    env.step([2, 2])
    
    env.step([5, 5])
    env.step([5, 5])
    env.step([5, 5])
    env.step([5, 5])
    
    
    for i in range(3):
        env.step([9, 9])
        env.step([3, 3])
        env.step([6, 6])
        env.step([2, 2])
        env.step([1, 1])
        env.step([6, 6])
        env.step([0, 0])
        env.step([4, 4])
    
    env.step((1, 0))
    env.step((1, 0))
    env.step((1, 0))
    env.step((1, 0))
    
    env.step((5, 0))
    env.step((5, 0))
    env.step((5, 0))
    env.step((5, 0))
    
    for i in range(3):
        env.step((9, 0))
        env.step((0, 0))
        env.step((6, 0))
        env.step((1, 0))
        env.step((3, 0))
        env.step((6, 0))
        env.step((2, 0))
        env.step((4, 0))
    
    env.step((3, 0))
    env.step((3, 0))
    env.step((3, 0))
    env.step((3, 0))
    
    env.step((5, 0))
    env.step((5, 0))
    env.step((5, 0))
    env.step((5, 0))
    
    for i in range(3):
        env.step((9, 0))
        env.step((0, 0))
        env.step((6, 0))
        env.step((1, 0))
        env.step((2, 0))
        env.step((6, 0))
        env.step((3, 0))
        env.step((4, 0))
    
    env.step((8, 0))
    env.step((0, 0))
    env.step((5, 0))
    #env.render()
    #print(env.building_zone)
    env.step((2, 0))
    env.step((6, 0))
    env.step((0, 0))
    env.step((6, 0))
    env.step((4, 0))
    env.step((3, 0))
    
    env.step((1, 0))
    env.step((8, 0))
    env.step((0, 0))
    """env.step((8, 0))
    
    env.step((0, 0))"""

    # forward increment the y
    # left increment the x
    # place BEAM
    env.step((Action.PLACE_BEAM, 0))
    # move forward
    env.step((Action.FORWARD, 0))
    # place Beam
    env.step((Action.PLACE_BEAM, 0))

    # move left 
    for i in range(2):
        env.step((Action.LEFT, 0))
        env.step((Action.PLACE_BEAM, 0))
    # move backward
    env.step((Action.BACKWARD, 0))
    # move down
    env.step((Action.DOWN, 0))
    #place scafold
    env.step((Action.PLACE_SCAFOLD, 0))
    # move backward and place scafold
    env.step((Action.BACKWARD, 0))
    env.step((Action.PLACE_SCAFOLD, 0))


    # move up and move backward
    env.step((Action.UP, 0))
    env.step((Action.BACKWARD, 0))

    for i in range(2):
        # move right 
        env.step((Action.RIGHT, 0))
    for i in range(3):
        # move left and place beam
        env.step((Action.LEFT, 0))
        env.step((Action.PLACE_BEAM, 0))
    for i in range(3):
        # move forward and place beam
        env.step((Action.FORWARD, 0))
        env.step((Action.PLACE_BEAM, 0))
    for i in range(1):
        # move right and place beam
        env.step((Action.RIGHT, 0))
        env.step((Action.PLACE_BEAM, 0))


    # move backward and move down and remove scafold
    env.step((Action.BACKWARD, 0))
    env.step((Action.DOWN, 0))
    # print  reward for removing scafold
    _, R, _, _, _ = env.step((Action.REMOVE_SCAFOLD, 0))
    print("isdone structure without scafold", env.unwrapped.finished_structure_with_scafold)
    print("reward is ", R)
    
    
if __name__ == "__main__":
    import time
    

    env = GridWorldEnv(6, path="targets" , batch_size=2)
    env.agent_pos_grid.fill(0)
    env.agent_pos = np.zeros((2, 3), dtype=int)
    print(env.agent_pos)
    obs, reward, _, _, _ = env.step((9, 9))
    print(reward)
    env.step((0, 0))
    env.step((3, 0))
    
    for _ in range(3):
        env.step((9, 0))
        env.step((1, 0))
        env.step((6, 0))
        env.step((0, 0))
        env.step((2, 0))
        env.step((6, 0))
        env.step((3, 0))
        env.step((0, 0))
        env.step((6, 0))
        env.step((1, 0))
        env.step((3, 0))
        env.step((6, 0))
        env.step((2, 0))
        env.step((4, 0))
    
    [env.step((3, 0)) for _ in range(3)]
    [env.step((5, 0)) for _ in range(4)]
    
    for _ in range(3):
        env.step((9, 0))
        env.step((1, 0))
        env.step((6, 0))
        env.step((0, 0))
        env.step((2, 0))
        env.step((6, 0))
        env.step((3, 0))
        env.step((0, 0))
        env.step((6, 0))
        env.step((1, 0))
        env.step((3, 0))
        env.step((6, 0))
        env.step((2, 0))
        env.step((4, 0))
    
    [env.step((0, 0)) for _ in range(3)]
    [env.step((5, 0)) for _ in range(4)]
    
    
    for _ in range(3):
        env.step((9, 0))
        env.step((1, 0))
        env.step((6, 0))
        env.step((0, 0))
        env.step((2, 0))
        env.step((6, 0))
        env.step((3, 0))
        env.step((0, 0))
        env.step((6, 0))
        env.step((1, 0))
        env.step((3, 0))
        env.step((6, 0))
        env.step((2, 0))
        env.step((4, 0))
    
    [env.step((2, 0)) for _ in range(3)]
    [env.step((5, 0)) for _ in range(4)]
    
    for _ in range(3):
        env.step((9, 0))
        env.step((1, 0))
        env.step((6, 0))
        env.step((0, 0))
        env.step((2, 0))
        env.step((6, 0))
        env.step((3, 0))
        env.step((0, 0))
        env.step((6, 0))
        env.step((1, 0))
        env.step((3, 0))
        env.step((6, 0))
        env.step((2, 0))
        env.step((4, 0))
    
    env.step((1, 0))
    env.step((5, 0))
    env.step((2, 0))
    for _ in range(3):
        env.step((6, 0))
        env.step((3,0))
        env.step((3, 0))
        env.step((6, 0))
        env.step((1, 0))
        env.step((2, 0))
        env.step((2, 0))
    
    for _ in range(6):
        env.step((6, 0))
        env.step((3, 0))
    for _ in range(6):
        env.step((6, 0))
        env.step((0, 0))
    for _ in range(6):
        env.step((6, 0))
        env.step((2, 0))
    
    [env.step((3, 0)) for _ in range(3)]
    env.step((1, 0))
    env.step((1, 0))
    env.step((6, 0))
    env.step((1, 0))
    env.step((6, 0))      
    env.step((4, 0))
    [env.step((2, 0)) for _ in range(2)]  
    
    env.step((1, 0))
    
    for _ in range(4):
        env.step((8, 0))
        env.step((0, 0))
    env.step((1, 0))
    for _ in range(4):
        env.step((8, 0))
        env.step((3, 0))
    env.step((2, 0))
    for _ in range(4):
        env.step((8, 0))
        env.step((1, 0))
    env.step((0, 0))
    for _ in range(4):
        env.step((8, 0))
        env.step((2, 0))
    
    env.step((5, 0))
    obs, reward, done, terminated, _ = env.step((7, 0))
    print(reward)
    env.step((0, 0))
    env.step((5, 5))
    env.step((5, 5))
    env.step((5, 5))
    _, reward, done, terminated, _ = env.step((8, 8))
    
    print(reward)
    [env.step((1, 1)) for _ in range(2)]
    [env.step((4, 4)) for _ in range(6)]
    
    for k in range(6):
        for j in range(6):
            for i in range(6):
                env.step((7, 7))
                env.step((0, 0))
                print("done", env.done_array)
                
            [env.step((1, 1)) for _ in range(6)]
            env.step((5, 5))
        [env.step((4, 4)) for _ in range(6)]
        env.step((3, 3))
    

    
    env.render()
    
    
    
    
