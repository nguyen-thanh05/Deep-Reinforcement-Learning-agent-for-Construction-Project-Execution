import time

from grid_world import GridWorldEnv
from grid_world_old import GridWorldEnvOld
from random import randint
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

TAU = 0.005


class DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DQN, self).__init__()
        self.in_conv = nn.Conv3d(3, 32, 3, 1, 1)

        self.conv1 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
        )

        self.fc1 = nn.Linear((32 + 3)*input_dim * input_dim * input_dim, 512)
        self.actions = nn.Linear(512, action_dim)
        self.advantage = nn.Linear(512, 1)
    def forward(self, x):
        original_state = x

        x = self.in_conv(x)

        x = x + self.conv1(x)

        x = F.relu(x)

        x = x + self.conv2(x)
        x = F.relu(x)

        x = x + self.conv3(x)
        x = F.relu(x)

        x = torch.cat([x, original_state], dim=1)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = F.relu(x)
        actions = self.actions(x)
        advantage = self.advantage(x)
        x = advantage + (actions - actions.mean())
        return x


def test_env_behavior():
    env_new = GridWorldEnv(8)
    env_old = GridWorldEnvOld(8)

    for _ in range(100):
        env_new.reset()
        env_old.reset()

        for __ in range(100):
            action = randint(0, 6)
            obs_new, reward_new, x, y, z = env_new.step(action)
            obs_old, reward_old, a, b, c = env_old.step(action)

            try:
                assert np.array_equal(obs_old, obs_new)
                assert reward_old == reward_new
            except AssertionError:
                print(obs_old[0], obs_new[0])
                print(obs_old[1], obs_new[1])

                print(reward_old, reward_new)
                break


def test_load_dict_performance():
    start = time.time()
    policy_net = DQN(4, 7)
    target_net = DQN(4, 7)

    policy_net.cuda()
    target_net.cuda()

    for _ in tqdm(range(200 * 300)):
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        # comment the line below out to measure time difference
        target_net.load_state_dict(target_net_state_dict)
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    # test_load_dict_performance()
    test_env_behavior()