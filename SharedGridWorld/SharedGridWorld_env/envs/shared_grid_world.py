import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch


from threading import Lock , Thread
MAX_TIMESTEP = 500

class SharedGridWorldEnv(gym.Env): 
    """

        4 TUPLE element state
        1. agent position
        2. other agents position
        3. building zone
        4. target zone

    """ 
    def __init__(self, dimension_size, num_agents):
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.dimension_size = dimension_size
        self.timestep_elapsed = 0
        self.mutex = Lock()
        self.num_agents = num_agents

        self.reset()
    
    def reset(self, seed=None, options=None):
        self.building_zone = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)

        self.AgentsPos = np.zeros((self.num_agents, 3), dtype=int)

        self.mutex.acquire()  # get lock to enter critical section
        
        random_start_pos = np.zeros(3, dtype=int)
        for i in range(self.num_agents):
            random_start_pos[0] = np.random.randint(0, self.dimension_size)
            random_start_pos[1] = np.random.randint(0, self.dimension_size)
            random_start_pos[2] = np.random.randint(0, self.dimension_size)
            self.AgentsPos[i] = random_start_pos
        #self.agent_pos = [random_start_pos[0], random_start_pos[1], random_start_pos[2]]
        
        # List of actions
        # 0: forward, 1: backward
        # 2: left, 3: right
        # 4: up, 5: down
        # 6: pick
        self.action_space = spaces.Discrete(7)   
        self._init_target()
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)

        self.timestep_elapsed = 0


        obs = self.get_obs(0)
        self.mutex.release()
        return obs, {}

    def getAgentPos(self, agent_id=0):
        return self.AgentsPos[agent_id]

    def get_obs(self, agent_id):
        # clear agent_pos_grid
        # TOOD: return the 4 TUPLES
        # 1. agent position
        # 2. other agents position
        # 3. building zone
        # 4. target zone

        agent_pos_grid = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        agent_pos_grid[self.AgentsPos[agent_id][0], self.AgentsPos[agent_id][1], self.AgentsPos[agent_id][2]] = 1


        other_agents_pos_grid = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        for i in range(self.num_agents):
            if i != agent_id:
                other_agents_pos_grid[self.AgentsPos[i][0], self.AgentsPos[i][1], self.AgentsPos[i][2]] = 1

        
        return np.stack((agent_pos_grid, other_agents_pos_grid ,self.building_zone ,self.target), axis=0)
        
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

    def step(self, actionTuple):



        action = actionTuple[0]
        agent_id = actionTuple[1]


        agent_pos = self.getAgentPos(agent_id)
        self.timestep_elapsed += 1
        move_cmd = False
        place_cmd = False
        # List of actions
        # 0: forward, 1: backward
        # 2: left, 3: right
        # 4: up, 5: down
        # 6: pick


        self.mutex.acquire()  # get lock to enter critical section

        if action == 1:
            # Y - 1
            if agent_pos[1] > 0:
                self.AgentsPos[agent_id][1] -= 1
            move_cmd = True
        elif action == 0:
            # Y + 1
            if agent_pos[1] < self.dimension_size - 1:
                self.AgentsPos[agent_id][1] += 1
            move_cmd = True
                
        elif action == 2:
            # X - 1
            if agent_pos[0] > 0:
                self.AgentsPos[agent_id][0] -= 1
            move_cmd = True
                
        elif action == 3:
            # X + 1
            if agent_pos[0] < self.dimension_size - 1:
                self.AgentsPos[agent_id][0] += 1
            move_cmd = True
        
        elif action == 4:
            # Z + 1
            if agent_pos[2] < self.dimension_size - 1:
                self.AgentsPos[agent_id][2] += 1
            move_cmd = True       
        elif action == 5:
            # Z - 1
            if agent_pos[2] > 0:
                self.AgentsPos[agent_id][2] -= 1
            move_cmd = True
        
        elif action == 6: # Place a block
            place_cmd = True
            # Find all 6 neighbouring directions
            neighbour_direction = [
                [agent_pos[0] + delta_x, agent_pos[1] + delta_y, agent_pos[2] + delta_z]
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
                if self.building_zone[agent_pos[0], agent_pos[1], agent_pos[2]] == 1:
                    supporting_neighbour = False
                # Place the block
                self.building_zone[agent_pos[0], agent_pos[1], agent_pos[2]] = 1
            else:
                # Check if block on the ground. No need to check support
                if agent_pos[2] == 0:
                    self.building_zone[agent_pos[0], agent_pos[1], agent_pos[2]] = 1
                    supporting_neighbour = True
        

        # return observation, reward, terminated, truncated, info
        if move_cmd:
            obs = self.get_obs(agent_id)
            self.mutex.release()
            if self.timestep_elapsed > MAX_TIMESTEP:
                return obs, torch.tensor(-0.2), torch.tensor(0), torch.tensor(1), {}
            else:
                return obs, torch.tensor(-0.2), torch.tensor(0), torch.tensor(0), {}
        elif place_cmd:
            obs = self.get_obs(agent_id)
            self.mutex.release()
            if supporting_neighbour:
                difference = self.target - self.building_zone
                difference = np.isin(difference, 1)



                if np.any(difference) == False:
                    return obs, torch.tensor(0), torch.tensor(1), torch.tensor(0), {}
                else:
                    if self.building_zone[agent_pos[0], agent_pos[1], agent_pos[2]] == self.target[agent_pos[0], agent_pos[1], agent_pos[2]]:
                        if self.timestep_elapsed > MAX_TIMESTEP:
                            return obs, torch.tensor(-0.2), torch.tensor(0), torch.tensor(1), {}
                        else:
                            return obs, torch.tensor(-0.2), torch.tensor(0), torch.tensor(0), {}
                    else:
                        if self.timestep_elapsed > MAX_TIMESTEP:
                            return obs, torch.tensor(-0.5), torch.tensor(0), torch.tensor(1), {}
                        else:
                            return obs, torch.tensor(-0.5), torch.tensor(0), torch.tensor(0), {}
            else:
                if self.timestep_elapsed > MAX_TIMESTEP:
                    return obs, torch.tensor(-0.5), torch.tensor(0), torch.tensor(1), {}        
                else:
                    return obs, torch.tensor(-0.5), torch.tensor(0), torch.tensor(0), {}

 
    def _render(self, agent_id):
        if (agent_id != 0): return # only master agent can render


        agent_pos = self.getAgentPos(0)
        agent_pos_grid = np.zeros((self.dimension_size, self.dimension_size, self.dimension_size), dtype=int)
        for i in range(self.num_agents):
            agent_pos_grid[self.AgentsPos[i][0], self.AgentsPos[i][1], self.AgentsPos[i][2]] = 1
        #agent_pos_grid[agent_pos[0], agent_pos[1], agent_pos[2]] = 1

        # prepare some coordinates
        targetCube = self.building_zone == 1
        agentCube = agent_pos_grid == 1




        cube1 = targetCube | agentCube
        # set the colors of each object
        colors = np.empty(cube1.shape, dtype=object)
        colors[targetCube] = '#7A88CCC0'
        colors[agentCube] = '#FFD65DC0'
        #print(colors)

        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(cube1, facecolors=colors, edgecolor='k')

        plt.show()
        return
    
    
    def close(self):
        pass


def testThread(env, agent_id):
    action_input = (0, 0)
    env.reset()
    env.step(action_input)
    action_input = (6, 0)
    env.step(action_input)
    action_input = (4, 0)
    s, _, _, _, _ = env.step(action_input)
    print("state shape is ",  s.shape)
    #env._render(0)
    print("DONE THREAD")
    return
    
if __name__ == "__main__":
    # List of actions
    # 0: forward, 1: backward
    # 2: left, 3: right
    # 4: up, 5: down
    # 6: place block
    env = SharedGridWorldEnv(5, 2)
    T = [Thread(target=testThread, args=(env, 0)), Thread(target=testThread, args=(env, 1))]
    print("START THREAD")
    for t1 in T:
        t1.start()


    for t2 in T:
        t2.join()
    print("JOIN THREAD")

    
