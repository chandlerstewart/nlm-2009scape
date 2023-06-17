
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import itertools
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import constants


plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        sample_list = self.sample_list()
        return random.sample(sample_list, batch_size)
    
    def sample_len(self):
        
        if len(self.memory) < constants.NUM_BOTS:
            return 0
        
        sample_len = len(list(itertools.islice(self.memory, 0, len(self.memory) - constants.NUM_BOTS)))
        return sample_len
    

    def sample_list(self):
        sample_list = list(itertools.islice(self.memory, 0, self.sample_len()))
        return sample_list
    
    def reward_mean(self):
        if len(self.memory) <= constants.NUM_BOTS:
            return 0
        
        rewards = torch.Tensor([t.reward for t in self.sample_list()])
        return torch.mean(rewards).item()
    
    def reward_max(self):
        if len(self.memory) <= constants.NUM_BOTS:
            return 0
        
        rewards = torch.Tensor([t.reward for t in self.sample_list()])
        return torch.max(rewards).item()
    
    def episode_reward_max(self):
        if len(self.memory) <= constants.NUM_BOTS:
            return 0
        
        rewards = torch.Tensor([t.reward for t in self.sample_list()])
        episode_rewards = rewards[rewards.shape[0] - constants.NUM_BOTS * constants.EPISODE_NUM_STEPS_MAX:]

        return torch.max(episode_rewards).item()
    
    def episode_reward_mean(self):
        if len(self.memory) <= constants.NUM_BOTS:
            return 0
        
        rewards = torch.Tensor([t.reward for t in self.sample_list()])
        episode_rewards = rewards[rewards.shape[0] - constants.NUM_BOTS * constants.EPISODE_NUM_STEPS_MAX:]

        return torch.mean(episode_rewards).item()
    
    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)
    
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        hidden_size = 128
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size,hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)





