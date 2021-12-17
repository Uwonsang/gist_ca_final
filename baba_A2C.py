from baba_is_gym import Env, time_measure, get_keyboard_input
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.functional as f
from torch.distributions import Categorical

import cv2
from matplotlib.pyplot import imshow, show

def pre_processing(state):
    # state = cv2.cvtColor(state,cv2.COLOR_BGR2RGB)
    state = np.dot(state[...,:3], [0.2989, 0.5870, 0.1140])
    state = cv2.resize(state,(84, 84), interpolation=cv2.INTER_NEAREST)
    state = state/state.max()
    state = state.reshape((1,1,84,84))
    
    return state


class Simple(nn.Module):
    def __init__(self, action_space, observation_space):
        super(Simple, self).__init__()
        self.actor = nn.Sequential(
            nn.Conv2d(1,8,(8, 8),stride=4,padding=2),
            nn.ReLU(),
            nn.Conv2d(8,16,(4, 4),stride=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,16,(3, 3),stride=1,padding=1),
            nn.Flatten(1, -1),
            nn.Linear(784, 256),
            nn.Tanh(),
            nn.Linear(256, 5)
        )

        self.critic = nn.Sequential(
            nn.Conv2d(1,8,(8, 8),stride=4,padding=2),
            nn.ReLU(),
            nn.Conv2d(8,16,(4, 4),stride=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,16,(3, 3),stride=1,padding=1),
            nn.Flatten(1, -1),
            nn.Linear(784, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.gamma = 0.9
        self.actor_adam = optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_adam = optim.Adam(self.critic.parameters(), lr=0.001)
    
    def forward(self, x):
        x = self.actor(x)
        x = nn.functional.softmax(x,dim = -1)
        # x = f.softmax(x,dim = -1)
        return x
        
    def get_action(self, state):
        val = self.forward(state)[0]
        m = Categorical(val)
        action = m.sample()
        return action, val[action]

    def inner_update(self, state, action_val, reward, next_state, done):
        self.actor_adam.zero_grad()
        self.critic_adam.zero_grad()
        next_action = self.forward(state)
        if done:
            target = reward
            adv = target -  self.critic(state)
        else : 
            target = reward + self.gamma * self.critic(next_state)
            adv = target - self.critic(state)
        critic_loss = (adv)**2
        actor_loss = -action_val * adv
        
        critic_loss.backward(retain_graph=True)
        self.critic_adam.step()
        actor_loss.backward()
        self.actor_adam.step()

        return critic_loss, actor_loss
    def update(self):
        pass

env = Env(3, training_on_single_stage=True) # 2 ~ 9
net = Simple(2,4)
epi_num = 1000
t = time_measure()
for i in range(epi_num):
    done = 0
    obs, _ = env.reset()
    obs = pre_processing(obs[0])
    r_sum = 0
    r = 0
    if i % 100 == 0:
            print('pass',i,'episodes')
    for i in range(200):
        t.start()
        obs = Variable(torch.tensor(obs).float())
        action, action_val = net.get_action(obs)
        env.render()
        obs_, reward, done, info = env.step(action.detach().numpy())
        r_sum += reward
        obs_ = pre_processing(obs_[0])
        obs_ = Variable(torch.tensor(obs_).float())
        net.inner_update(obs,action_val, reward, obs_, done)
        obs = obs_
        r += t.end()

        print(r/(i+1))
        if done:
            break
            