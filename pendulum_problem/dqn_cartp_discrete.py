import gym
import numpy as np
import argparse
from itertools import count
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

fname = 'cartpole_discrete_02'

last_score_plot = [0]
avg_score_plot = [0]

device = torch.device("cuda:02")

parser = argparse.ArgumentParser(description='PyTorch DQN solution of CartPole-v0')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--epsilon_start', type=float, default=0.9)
parser.add_argument('--epsilon_end', type=float, default=0.05)
parser.add_argument('--target_update', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--max_episode', type=int, default=500)

cfg = parser.parse_args()

env = gym.make('CartPole-v0')


class Memory(object):
    def __init__(self, memory_size=10000):
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size

    def __len__(self):
        return len(self.memory)

    def append(self, item):
        self.memory.append(item)

    def sample_batch(self, batch_size):
        idx = np.random.permutation(len(self.memory))[:batch_size]
        return [self.memory[i] for i in idx]


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = self.fc3(x)
        return x


def get_action(state, epsilon):
    with torch.no_grad():
        greedy_action = torch.argmax(policy_net(state), dim=1).item()
        random_action = np.random.randint(0, 2)
        return random_action if np.random.rand() < epsilon else greedy_action


def update_network(states, actions, next_states, rewards, dones):
    state_action_values = policy_net(states).gather(1, actions[:, None].long()).squeeze()
    next_state_values = torch.max(target_net(next_states), dim=1)[0].detach()
    expected_state_action_values = rewards + next_state_values * (1 - dones) * cfg.gamma

    loss = F.mse_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.RMSprop(policy_net.parameters(), lr=cfg.lr, weight_decay=1e-4)
memory = Memory(10000)

print(cfg.max_episode)

for i in range(cfg.max_episode):
    episode_durations = 0
    state = env.reset()
    epsilon = (cfg.epsilon_end - cfg.epsilon_start) * (i / cfg.max_episode) + cfg.epsilon_start

    for t in count():
        action = get_action(torch.tensor(state).float()[None, :].to(device), epsilon)
        next_state, reward, done, _ = env.step(action)

        memory.append([state, action, next_state, reward, done])
        state = next_state

        if len(memory) > cfg.batch_size:
            states, actions, next_states, rewards, dones = \
            map(lambda x: torch.tensor(x).float().to(device), zip(*memory.sample_batch(cfg.batch_size)))

            update_network(states, actions, next_states, rewards, dones)

        if done:
            episode_durations = t + 1
            avg_score_plot.append(avg_score_plot[-1] * 0.99 + episode_durations * 0.01)
            last_score_plot.append(episode_durations)
            break

    # Update the target network
    if i % cfg.target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print(i)
    
print('Complete')
env.close()

import pickle
pickle.dump([last_score_plot, avg_score_plot], open(fname+'.pkl', 'wb'))
torch.save(target_net.state_dict(), fname+'_target.pt')
torch.save(policy_net.state_dict(), fname+'_policy.pt')

