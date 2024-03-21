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

        return dist, value

import torchinfo
test = ActorCritic(4, 8)
torchinfo.summary(test, (1, 3, 4, 4, 4))


def plot_durations(show_result=False):
    plt.figure(1)

    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training")
    plt.xlabel("Episode")
    plt.ylabel("Duration")

    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

        plt.pause(0.001)
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())


def optimise_model():
    if len(memory) < BATCH_SIZE * 64:
        return 0, 0
    transitions = memory.sample_batch()
    batch = Transition(
        torch.tensor(transitions["obs"], device=device),
        torch.tensor(transitions["acts"], device=device, dtype=torch.int64),
        torch.tensor(transitions["next_obs"], device=device),
        torch.tensor(transitions["rews"], device=device),
        torch.tensor(transitions["done"], device=device)
    )

    state_batch = batch.state
    action_batch = batch.action
    reward_batch = batch.reward
    next_state_batch = batch.next_state
    done_batch = batch.done

    # Compute V(s) and pi(a|s) for the current state batch
    dist, value = model(state_batch)

    # Compute V(s') for the next state batch
    _, next_value = model(next_state_batch)

    # calculate TD error = r + gamma * V(s') - V(s)
    # if terminal state, V(s') = 0
    td_target = reward_batch + GAMMA * next_value * (1 - done_batch) - value

    # critic loss = TD^2
    critic_loss = (td_target ** 2).mean()

    # calculate the log probabilities of the action batch
    log_probs = dist.log_prob(action_batch)

    # actor loss is the log probability of the action multiplied by the TD error
    actor_loss = -(log_probs * td_target.detach()).mean()

    # total loss
    loss = critic_loss + actor_loss

    return loss.item(), reward_batch.float()


STEPSIZE = 0.0000625
GAMMA = 0.9
N_STEP = 1
BATCH_SIZE = 32

memory = ReplayBuffer(obs_dim=(3,4,4,4), size=8192, n_step=N_STEP, gamma = GAMMA)

model = ActorCritic(4, 8)
optimiser = optim.Adam(model.parameters(), lr=STEPSIZE, eps=1.5e-4)

env = gym.make("GridWorld_env/GridWorld", dimension_size=4, path="targets")
env.reset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if torch.cuda.is_available():
    num_episodes = 1000
else:
    num_episodes = 10

episode_durations = []
reward_plot = []

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    cumulative_reward = 0

    for t in count():
        # get V(s) and pi(a|s) for the current state
        dist, value = model(state)

        # sample action from the distribution
        action = dist.sample()

        # take the action, observe R, S'
        next_state, reward, terminated, truncated, _ = env.step(action.item())

        cumulative_reward += reward
        reward = torch.tensor([reward], device=device)

        done = terminated or truncated

        next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.store(state.cpu().numpy(), action, reward.__float__(),
                     next_state.cpu().numpy(), terminated)
        state = next_state

        l, r = optimise_model()

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

    if i_episode % 1 == 0 and i_episode > 1:
        print("Episode: {0} Loss {1} Mean Sample Reward {2}:".format(i_episode, l, r.mean().item()))
        env.unwrapped.render()

    reward_plot.append(cumulative_reward)

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()

