# %%
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
import GridWorld_env
from replay_buffer import ReplayBuffer
import gymnasium as gym
import random
import math
from itertools import count
import __main__
device = "cuda" if torch.cuda.is_available() else "cpu"
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    
from collections import deque, namedtuple
import os

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# %%
# for vanilla and double DQN
class Vanilla_DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Vanilla_DQN, self).__init__()
               
        self.conv1 = nn.Conv3d(3, 29, 3, 1, 1)
        
        self.conv2 = nn.Conv3d(32, 67, 3, 1, 1)
        
        self.fc1 = nn.Linear((64 + 3*2)*input_dim * input_dim * input_dim, 1024)
        
        self.actions = nn.Linear(1024, action_dim)
        self.advantage = nn.Linear(1024, 1)
    def forward(self, x):
        original_state = x
               
        x = self.conv1(x)
        
        x = torch.cat([x, original_state], dim=1)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = torch.cat([x, original_state], dim=1)
        
        x = F.relu(x)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = F.relu(x)
        actions = self.actions(x)
        
        return actions

class Dueling_DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Dueling_DQN, self).__init__()
               
        self.conv1 = nn.Conv3d(3, 29, 3, 1, 1)
        
        self.conv2 = nn.Conv3d(32, 67, 3, 1, 1)
        
        self.fc1 = nn.Linear((64 + 3*2)*input_dim * input_dim * input_dim, 1024)
        
        self.actions = nn.Linear(1024, action_dim)
        self.advantage = nn.Linear(1024, 1)
    def forward(self, x):
        original_state = x
               
        x = self.conv1(x)
        
        x = torch.cat([x, original_state], dim=1)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = torch.cat([x, original_state], dim=1)
        
        x = F.relu(x)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = F.relu(x)
        actions = self.actions(x)
        advantage = self.advantage(x)
        x = advantage + (actions - actions.mean(dim=1, keepdim=True))

        #print("advantage", advantage)
        #print("Value ", x)
        return x
    
#LINEAR MODEL
#class Vanilla_DQN(nn.Module):
#    def __init__(self, input_dim, action_dim):
#        super(Vanilla_DQN, self).__init__()
#
#        self.fc1 = nn.Linear(192, 1024)
#        self.fc2 = nn.Linear(1024, 1024)
#        self.fc3 = nn.Linear(1024, 1024)
#        self.fc4 = nn.Linear(1024, 1024)
#        self.fc5 = nn.Linear(1024, 1024)
#
#        self.actions = nn.Linear(1024, action_dim)
#
#    def forward(self, x):
#        x = nn.Flatten()(x)
#
#        x = self.fc1(x)
#        x = F.relu(x)
#
#        x = x + self.fc2(x)
#        x = F.relu(x)
#
#        x = x + self.fc3(x)
#        x = F.relu(x)
#
#        x = x + self.fc4(x)
#        x = F.relu(x)
#
#        x = x + self.fc5(x)
#        x = F.relu(x)
#
#        actions = self.actions(x)
#
#        return actions

# %%
n_actions = 8
env = gym.make("GridWorld_env/GridWorld", dimension_size=4, path="targets")
env.reset()

FOLDER_NAME = str(input("Enter the folder name: ")) # this isstring


if ("Dueling" in FOLDER_NAME):
    print("DUELING MODEL")
    policy_net = Dueling_DQN(4, 8)
    target_net = Dueling_DQN(4, 8)
elif ("Linear" in FOLDER_NAME):
    print("LINEAR MODEL")
    policy_net = Vanilla_DQN(4, 8)
    target_net = Vanilla_DQN(4, 8)
else:
    print("VANILLA MODEL")
    policy_net = Vanilla_DQN(4, 8)
    target_net = Vanilla_DQN(4, 8)

#%%
#target_net = torch.load("../dqn_models/vanilla/vanilla_dqn_target.pt")


# %%        return policy_net(state).max(1).indices.view(1,1)
eps_threshold = 0.01
def select_action(state, greedy = False):
    
        
    if np.random.rand() > eps_threshold:
        with torch.no_grad():
            #print(policy_net(state))
            return policy_net(state).max(1).indices.view(1,1)
    else:
        #print(policy_net(state))
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    

# %%


def getAllFileInFolder(folder_name):
    # function that create a list of all files in this folder
    try:
        # Get the list of files in the specified directory
        files = os.listdir(folder_name)

        return files
    except OSError:
        # Handle the case where the folder doesn't exist or there's a permission issue
        print(f"Error: Unable to list files in '{folder_name}'")
        return []




files_list = getAllFileInFolder(FOLDER_NAME + "/Weights")


print(files_list)
stat_dict = dict()

result_seq = np.zeros((10, 10))
result_move = np.zeros((10, 10))
result_block = np.zeros((10, 10))


# TODO: remove this for weights
#state, info = env.reset()
#policy_net = torch.load(FOLDER_NAME + "/Weights/1.pt" , map_location=torch.device("cpu"))
#state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
#action = select_action(state)  # prints
#exit()

for i in range(10):
    for j, file in enumerate(files_list):
        
        policy_net = torch.load(FOLDER_NAME + "/Weights/" + file , map_location=torch.device("cpu"))
        seq = []
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state, greedy = True)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            state = next_state
            if terminated or truncated:
                break
            #env.unwrapped.render()
            seq.append(action.item())
            #display.clear_output(wait=True)


        state, inf = env.reset()

        # count the
        # conver seq to NP array
        seq = np.array(seq)
        result_seq[i, j] = len(seq)
        result_block[i, j] = np.sum(seq >= 6)
        result_move[i, j] = np.sum(seq < 6)
        # count the number of elements less than 6
        print("number of moves in and number blocks placed in ", FOLDER_NAME,"/", file, " agent is ",  np.sum(seq < 6), " ", np.sum(seq == 6) + np.sum(seq == 7))
        # count the number of action == 6
    #print(len(seq))


np.save("NumStepPrio/" + FOLDER_NAME + "_seq.npy", result_seq)
np.save("NumBlockPrio/" + FOLDER_NAME + "_block.npy", result_block)
np.save("NumMovePrio/" + FOLDER_NAME + "_move.npy", result_move)
#for a in seq:
#    state, reward, terminated, truncated, _ = env.step(a)
#    env.unwrapped.render()
#    display.clear_output(wait=True)

exit()
import matplotlib.pyplot as plt
# bin edge where 0 to 120 is incremented by 5 and rest is incremented by 100 up to 800


plt.figure(figsize=(10, 5))
plt.xticks(np.arange(0, 900, 100))
bin_edges = [i for i in range(0, 900, 5)]   # good for vanilaDQN
#bin_edges = [i for i in range(0, 900, 100)]   # good for Linear 

#bin_edges = [i for i in range(0, 150, 5)] + [i for i in range(150, 900, 50)]
#bin_edges = [i for i in range(0, 20, 5)] + [i for i in range(20, 40, 2)] +  [i for i in range(40, 150, 5)] + [i for i in range(150, 900, 50)]
#plt.hist(np.resize(result_seq, (100,)) , bins = bin_edges, edgecolor='black', color='red')
plt.hist(np.resize(result_seq, (100,)), bins=bin_edges, edgecolor='black', color='purple')

# add title of FILE_NAME number of sequence
#
plt.title(FOLDER_NAME + ": number of steps", fontsize=24)
# add x axis name as duration

#plt.xlim(0, 150)
plt.xlabel("number of steps")
plt.ylabel("Frequency")
plt.legend(['Median = {:.2f}'.format(np.median(result_seq))])
#plt.show()

#np.save("NumStep/" + FOLDER_NAME + "_seq.npy", result_seq)

#exit()
# ---------------------------------------------------
bin_edges_block = [i for i in range(0, 300, 10)]   # good for vanilaDQN

plt.hist(np.resize(result_block, (100, )), edgecolor='black')

# add title of FILE_NAME number of sequence
plt.title("Histogram of " +  FOLDER_NAME +  ": number of blocks placed")
# add x axis name as duration

plt.xlabel("number of block placed")
plt.ylabel("Frequency")
plt.legend(['Median = {:.2f}'.format(np.median(result_block))])
plt.show()


#plt.hist(result_move)
print("mean numseq ", np.mean(result_seq))
print("median numseq", np.median(result_seq))
print("mean move ", np.mean(result_move))
print(np.mean(np.mean(result_block)))


