from typing import final
import GridWorld_env
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from itertools import count
from replay_buffer import ReplayBuffer
import matplotlib
from collections import namedtuple

is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))


class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.conv1 = nn.Conv3d(3, 29, 3, 1, 1)

        self.conv2 = nn.Conv3d(32, 67, 3, 1, 1)

        self.fc1 = nn.Linear((64 + 3*2)*input_dim * input_dim * input_dim, 1024)

        self.critic = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )

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

        value = self.critic(x)
        probs = self.actor(x)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
            
        return action, log_prob, entropy, value.squeeze(0), None


class RolloutStorage(object):
    def __init__(self, rollout_size, num_envs, feature_size, is_cuda=True, value_coeff=0.5, entropy_coeff=0.2, writer=None):
        super().__init__()
        
        self.rollout_size = rollout_size
        self.num_envs = num_envs
        self.feature_size = feature_size
        self.is_cuda = is_cuda
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.writer = writer
        
        self.rewards = self._reset_buffer_with_zero((rollout_size, num_envs))
        self.states = self._reset_buffer_with_zero((rollout_size + 1, num_envs, 3, feature_size, feature_size, feature_size))
        self.actions = self._reset_buffer_with_zero((rollout_size, num_envs))
        self.log_probs = self._reset_buffer_with_zero((rollout_size, num_envs))
        self.values = self._reset_buffer_with_zero((rollout_size, num_envs))
        self.dones = self._reset_buffer_with_zero((rollout_size, num_envs))
        
    
    def _reset_buffer_with_zero(self, size):
        if self.is_cuda:
            return torch.zeros(size).cuda()
        return torch.zeros(size)
    
    def insert(self, step, reward, obs, action, log_prob, value, dones):
        self.rewards[step].copy_(torch.tensor(reward))
        self.states[step+1].copy_(torch.from_numpy(obs))
        self.actions[step].copy_(action)
        self.log_probs[step].copy_(log_prob)
        self.values[step].copy_(value)
        
        self.dones[step].copy_(dones)
        
    def _discount_rewards(self, final_value, discount=0.99):
        r_discounted = self._reset_buffer_with_zero((self.rollout_size, self.num_envs))
        R = self._reset_buffer_with_zero(self.num_envs).masked_scatter((1 - self.dones[-1]).byte(), final_value)
        #R = R.to(self.rewards.device)
        for i in reversed(range(self.rollout_size)):
            #print(self.dones[i].device, self.rewards[i].device, R.device)
            R = self._reset_buffer_with_zero(self.num_envs).masked_scatter((1 - self.dones[i]).byte(), self.rewards[i] + discount * R)
            r_discounted[i] = R 
            
        return r_discounted
    
    def compute_a2c_loss(self, final_value, entropy):
        rewards = self._discount_rewards(final_value)
        advantages = rewards - self.values
        
        policy_loss = (-self.log_probs * advantages.detach()).mean()
        
        value_loss = advantages.pow(2).mean()
        
        loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
        
        return loss

    def get_state(self, step):
        return self.states[step].clone()
    
    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.actions = self._reset_buffer_with_zero((self.rollout_size, self.num_envs))
        self.log_probs = self._reset_buffer_with_zero((self.rollout_size, self.num_envs))
        self.values = self._reset_buffer_with_zero((self.rollout_size, self.num_envs))
        
    


class Runner:
    def __init__(self, net, env, num_envs, rollout_size, num_updates, max_grad_norm, value_coeff=0.5, entropy_coeff=0.02, tensorboard_log = False, log_path="./log", is_cuda = True):
        self.num_envs = num_envs
        self.rollout_size = rollout_size
        self.num_updates = num_updates
        
        self.max_grad_norm = max_grad_norm
        
        self.is_cuda = torch.cuda.is_available() and is_cuda
        self.writer = None
        
        
        self.env = env
        self.storage = RolloutStorage(rollout_size, num_envs, 4, self.is_cuda, value_coeff, entropy_coeff, self.writer)
        
        self.net = net
        
        if self.is_cuda:
            self.net.cuda()
            
    def train(self, optimizer):
        self.env.reset()
        obs = self.env.get_obs()
        self.storage.states[0].copy_(torch.from_numpy(obs))
        best_loss = np.inf
        
        
        for num_update in range(self.num_updates):
            final_value, episode_entropy = self.episode_rollout()
            opt.zero_grad()
            
            loss = self.storage.compute_a2c_loss(final_value, episode_entropy)
            loss.backward()
            
            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            opt.step()
            
            self.storage.after_update()
            
            if abs(loss) < abs(best_loss):
                best_loss = loss.item()
                print("model saved with best loss: ", best_loss, " at update #", num_update)
                torch.save(self.net.state_dict(), "a2c_best_loss")

            elif num_update % 1 == 0:
                print("current loss: ", loss.item(), " at update #", num_update)
                #self.storage.print_reward_stats()

            elif num_update % 1 == 0:
                torch.save(self.net.state_dict(), "a2c_time_log_no_norm")
            
        
        
    def episode_rollout(self):
        episode_entropy = 0
        for step in range(self.rollout_size):
            
            action, log_prob, entropy, value, a2c_features = self.net(self.storage.get_state(step))
            episode_entropy += entropy.item()
            #print(action, log_prob, entropy, value, a2c_features)
            obs, reward, terminated, dones, _ = self.env.step(action.cpu().numpy())
            #print(dones)
            self.storage.insert(step, reward, obs, action, log_prob, value, dones)
        
        with torch.no_grad():
            _, _, _, final_value, _ = self.net(self.storage.get_state(step + 1))
        return final_value, episode_entropy


n_actions = 8
env = gym.make("GridWorld_env/GridWorld", dimension_size=4, path="targets")
net = ActorCritic(4, n_actions)
runner = Runner(net=net, env=env, num_envs=1, rollout_size=1500, num_updates=1000, max_grad_norm=0.5, value_coeff=0.5, entropy_coeff=0.02, tensorboard_log=False, log_path="./log", is_cuda=True)

opt = optim.Adam(net.parameters(), lr=0.0001)
runner.train(opt)

env.reset()
for i in range(1500):
    
    action, _, _, _, _ = net(torch.from_numpy(env.get_obs()).unsqueeze(0).float().cuda())
    env.step(action.cpu().numpy())
env.render()
    