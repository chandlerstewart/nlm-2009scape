import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import constants
from QLearning import DQN, ReplayMemory, Transition
import server
from utils import State
import utils
import traceback

from Trainer import DoubleQTrainer

matplotlib.use("TkAgg")
plt.ion()
#plt.show(block=False)



Server = server.Server()
Server.start()

episode = 0
steps_this_episode = 0
steps_per_episode = constants.EPISODE_NUM_STEPS_MAX


trainer = DoubleQTrainer()

best_reward_mean = 0
mean_rewards = []

while episode < constants.NUM_EPISDOES:
    try:
        if Server.MESSAGE_IN_UPDATED:
                Server.update_message()

                if Server.STATE == State.SEND_ACTION:
                    trainer.step(Server.last_response)
                    Server.step(utils.bot_to_json(trainer.bots))
                    steps_this_episode += 1
                    
                    

                if steps_this_episode >= steps_per_episode:
                    episode += 1
                    steps_this_episode = 0
                    steps_per_episode = min(constants.EPISODE_NUM_STEPS_MAX, steps_per_episode + 5)


                    Server.STATE = State.RESET_EPISDOE
                    
                    
                    


    except Exception as e:
            traceback.print_exc()
            Server.close()
            exit(1)






