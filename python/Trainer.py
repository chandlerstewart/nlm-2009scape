import torch
from QLearning import *
from utils import *

import traceback




class DoubleQTrainer:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(constants.STATE_SIZE, constants.ACTION_SIZE).to(self.device)
        self.target_net = DQN(constants.STATE_SIZE, constants.ACTION_SIZE).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.batch_size = 256
        self.gamma = 0.99
        self.eps_start = 1
        self.eps_end = 0.01
        self.eps_decay = 100000
        self.tau = 0.005
        self.lr = 1e-4

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(20000)

        self.logs_collected = [0] * constants.NUM_BOTS
        self.mean_logs_collected = []

        self.steps_done = 0

        


        

    def optimize_model(self):
        if self.memory.sample_len() < self.batch_size:
            return
        transitions=self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        
        #print(batch.state)
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.stack(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.stack(batch.reward).to(self.device)
        next_states_batch = torch.stack(batch.next_state).to(self.device)


        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values = torch.max(self.target_net(next_states_batch), dim=1).values
            #next_state_values[non_final_mask] = self.target_net(next_states_batch).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)


    def select_action(self,state):
    
        sample = random.random()
        
        if sample > self.epsilon():
            with torch.no_grad():
                return torch.argmax(self.policy_net(state))
        else:
            return torch.tensor(random.randrange(0,constants.ACTION_SIZE), device=self.device, dtype=torch.long)
        
    def step(self, jsondata):
        if isinstance(jsondata, Message):
            print(traceback.print_stack())
            return
        
        self.bots = json_to_bot(jsondata) 
        num_bots = len(self.bots)

        for i in range(num_bots):
            bot=self.bots[i]

            state = torch.Tensor(bot.get_absolute_state()).to(self.device)
            action = self.select_action(state)
            bot.take_action(action)

            if (len(self.memory) >= num_bots):
                last_memory_index = len(self.memory) - num_bots
                last_state = self.memory.memory[last_memory_index].state
                last_action = self.memory.memory[last_memory_index].action
                last_reward = torch.as_tensor(bot.info["reward"]) + self.movement_reward(last_state, state)
                self.memory.memory[last_memory_index] = Transition(last_state, last_action, state, last_reward)


                freeInvSpace = bot.info["freeInvSpace"]
                lastFreeInvSpace = self.memory.memory[last_memory_index].state[2]

                if freeInvSpace < lastFreeInvSpace:
                    self.logs_collected[i] += lastFreeInvSpace - freeInvSpace
                

            self.memory.push(state, action, None, None)
        self.optimize_model()
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau +  target_net_state_dict[key]*(1-self.tau)

        self.target_net.load_state_dict(target_net_state_dict)

        self.steps_done += 1

    def save_model(self):
        print("Saving model")
        torch.save(self.policy_net.state_dict(), constants.MODEL_SAVE_PATH)


    def movement_reward(self, last_state, state):
        lastx, lasty = Bot.unnormalize_xy((last_state[0], last_state[1]))

        x,y = Bot.unnormalize_xy((state[0], state[1]))
        inv = state[2]

        last_dist = math.sqrt((lastx - constants.GOAL_LOC[0])**2 + (lasty - constants.GOAL_LOC[1])**2)
        dist = math.sqrt((x - constants.GOAL_LOC[0])**2 + (y - constants.GOAL_LOC[1])**2)

        if dist < last_dist:
            return 28 - inv
        else:
            return 0
        
    def clear_logs_collected(self):
        self.mean_logs_collected.append(torch.Tensor(self.logs_collected).mean().item())
        self.logs_collected = [0] * constants.NUM_BOTS
        

    def clear_memory(self):
        self.memory.clear()


    def reward_mean(self):
        return self.memory.episode_reward_mean()
    
    def reward_max(self):
        return self.memory.episode_reward_max()





