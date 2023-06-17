import torch
import constants
import json
from enum import Enum

import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from IPython import display
import numpy as np
import random


def one_hot_state(xystate):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.zeros(constants.X_SIZE, constants.Y_SIZE)
        state[xystate[0], xystate[1]] = 1
        state = state.flatten().to(device)
        return state

class Message:
    def __init__(self, command, info = ""):
        self.command = command
        self.info = info

    def to_json_out(self):
        json_out = json.dumps(self.__dict__)
        return json_out.encode()
    

class State(Enum):
    WAIT_FOR_CONNECTION = 0
    SPAWN_BOTS = 1
    WAIT_FOR_DATA = 2
    SEND_ACTION = 3
    RESET_EPISDOE = 4

def encode_state(state):
    #ret = state[0] * constants.X_SIZE + state[1]
    ret = [state[0], state[1]]
    return ret

def plot_rewards(rewards, show_result=False):
    
    plt.figure(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 20:
        means = rewards_t.unfold(0, 20, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(19), means))
        plt.plot(means.numpy())

    plt.pause(0.5)  # pause a bit so that plots are updated

    plt.savefig("./plots/rewards.png")

    if not show_result:
        display.display(plt.gcf())
        display.clear_output(wait=True)
    else:
        display.display(plt.gcf())

def plot_logs_collected(logs, show_result=False):
    
    plt.figure(2)
    logs_t = torch.tensor(logs, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Mean Logs Collected')
    plt.plot(logs_t.numpy())
    # Take 100 episode averages and plot them too
    if len(logs_t) >= 20:
        means = logs_t.unfold(0, 20, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(19), means))
        plt.plot(means.numpy())

    plt.pause(0.5)  # pause a bit so that plots are updated

    plt.savefig("./plots/logs_collected.png")

    if not show_result:
        display.display(plt.gcf())
        display.clear_output(wait=True)
    else:
        display.display(plt.gcf())
    

def json_to_bot(json_data):
    bots = []

    for val in json_data:
        bots.append(Bot(val))


    return bots

def bot_to_json(bots):
    return json.dumps([b.info for b in bots])


class Bot:



    def __init__(self, info):
        self.info = info

    def move_north(self):
        self.info["yLoc"] +=  1

    def move_south(self):
        self.info["yLoc"] -= 1

    def move_west(self):
        self.info["xLoc"] -= 1

    def move_east(self):
        self.info["xLoc"] += 1

    def get_absolute_state(self):
        state = [self.info["xLoc"],self.info["yLoc"]]
        state = self.normalize_xy(state)
        state += [self.info["freeInvSpace"]]
        state += self.nearby_nodes_to_state()
        
        return state
    
    def get_relative_state(self):
        state = [self.get_relative_x(),self.get_relative_y()]
        #state += self.get_state_bools()
        return state
    
    def unnormalize_xy(xy):
        x = xy[0] * constants.X_SIZE + constants.BOUNDSX[0]
        y = xy[1] * constants.Y_SIZE + constants.BOUNDSY[0]
        return [x,y]
    
    def normalize_xy(self, xy):
        x = (xy[0] - constants.BOUNDSX[0]) / constants.X_SIZE
        y = (xy[1] - constants.BOUNDSY[0]) / constants.Y_SIZE
        #print([x,y])
        return [x,y]

    def nearby_nodes_to_state(self):
        nodes = []
        for node in self.info["nearbyNodes"]:
            if node in ["", "Daisies"]:
                nodes.append(0)
            else:
                nodes.append(1)

        return nodes
        
    
    def take_action(self, action):
        if action in [0,1,2,3]:
            self.info["interact"] = "none"
        if action == 0:
            self.move_north()
        if action == 1:
            self.move_south()
        if action == 2:
            self.move_east()
        if action == 3:
            self.move_west()
        if action == 4:
            self.info["interact"] = "north"
        if action == 5:
            self.info["interact"] = "south"
        if action == 6:
            self.info["interact"] = "east" 
        if action == 7:
            self.info["interact"] = "west"

    def get_state_bools(self):
        bools = [
                 self.info["canMoveNorth"],self.info["canMoveSouth"], self.info["canMoveEast"], self.info["canMoveWest"],
                 self.info["northNode"], self.info["southNode"], self.info["eastNode"], self.info["westNode"]
                 ]
        

        bools = np.asarray(bools).astype(int)

        return bools.tolist()


    def random_move(self):
        choice = random.randint(0,3)

        if choice == 0:
            self.move_north()
        if choice == 1:
            self.move_south()
        if choice == 2:
            self.move_east()
        if choice == 3:
            self.move_west()

    def get_relative_x(self):
        return self.info["xLoc"] - constants.BOUNDSX[0]
    
    def get_relative_y(self):
        return self.info["yLoc"] - constants.BOUNDSY[0]

