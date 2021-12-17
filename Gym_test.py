from baba_is_gym import Env, get_keyboard_input
import cv2
import numpy as np

import collections
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import copy
import gym

epsilon = 0.9
max_episode = 300
memory_size = 2000
start_training = 256
batch_size = 32
gamma = 0.9
target_update_step = 400
learning_rate = 0.01

class DQN(nn.Module):
    def __init__(self, observation_space, action_space):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(observation_space,128)
        self.fc2 = nn.Linear(128,action_space)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        out = self.fc2(x)
        return out

    def get_action(self, obs):
        obs = to_tensor(obs)
        action = self.forward(obs)
        if epsilon > np.random.uniform():
            action = to_tensor(np.random.rand(2))
        action = torch.argmax(action)
        return int(action.detach().numpy())

def to_tensor(val):
    tensor_val = torch.tensor(val).float()
    return tensor_val

def train(main_net, target_net, batch, optimizer):
    obss, actions, rewards, obss_, dones = zip(*batch)

    obss = torch.tensor(obss).float()
    actions = torch.tensor(np.reshape(actions,(-1,1))).long()
    rewards = torch.tensor(rewards).float()
    obss_ = torch.tensor(obss_).float()
    dones = torch.tensor(dones).float()

    main_Q = main_net(obss).gather(1, actions).reshape(-1)
    target_Q = target_net(obss_).detach()

    y = rewards + gamma * target_Q.max(1)[0] * dones
    
    loss = nn.MSELoss()(main_Q, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    global epsilon
    env = gym.make("CartPole-v1")
    model = DQN(env.observation_space.shape[0], env.action_space.n)
    target = copy.deepcopy(model)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    replay_memory = [None] * memory_size
    idx = 0

    for i in range(max_episode):
        done = 0
        obs = env.reset()
        t_r = 0
        # obs = to_tensor(obs)
        while(not done):
            with torch.no_grad():
                action = model.get_action(obs)
                obs_, r, done, info = env.step(action)
                # obs_ = to_tensor(obs_)
                replay_memory[idx%memory_size] = [obs, action, r, obs_,not done]
                
                epsilon *= 0.99
                idx += 1
                t_r += r
                obs = obs_

            if idx >= start_training:
                if idx % target_update_step == 0:
                    target = copy.deepcopy(model)
                train(model, target, random.sample(replay_memory[:idx], batch_size), optimizer)
        
        
        print("episode {} total reward : {}".format(i, t_r))
    env.close()

if __name__ == '__main__':
    main()