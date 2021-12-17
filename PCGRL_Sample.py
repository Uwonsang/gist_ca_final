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

# 'O_empty':0
# 'O_baba':1
# 'O_rock':2
# 'O_flag':3
# 'O_wall':4
# 'O_skull'5
# 'O_lava':6
# 'O_water':7

# 'T_is':100
# 'T_baba':101
# 'T_rock':102
# 'T_flag':103
# 'T_wall':104
# 'T_skull':105
# 'T_lava':106
# 'T_water':107

# 'TR_you':200
# 'TR_win':201
# 'TR_stop':202
# 'TR_push':203
# 'TR_defeat':204
# 'TR_sink':205
# 'TR_hot':206
# 'TR_melt':207

# total 24
action_bag = [0,1,2,3,4,5,6,7,100,101,102,103,104,105,106,107,200,201,202,203,204,205,206,207]

pcg_seed = [
[[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0, 101, 100, 200,   0,   0,   0, 100, 201,   0],
 [  0,   4,   4,   4,   4,   4,   4,   4,   4,   0],
 [  0,   0,   0,   0,   2,   2,   0,   0,   0,   0],
 [  0,   0,   1,   0,   2,   2,   0,   3,   0,   0],
 [  0,   0,   0,   0,   2,   2,   0,   0,   0,   0],
 [  0,   4,   4,   4,   4,   4,   4,   4,   4,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0, 104, 100, 202,   0,   0, 102, 100, 203,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],

[[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0, 101, 100, 200,   0,   0, 103, 100, 201,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],

[[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0, 101, 100, 200,   0,   0, 0, 100, 201,   0],
 [  0,   4,   4,   4,   4,   4,   4,   4,   4,   0],
 [  0,   0,   0,   0,   2,   2,   0,   0,   0,   0],
 [  0,   0,   1,   0,   2,   2,   0,   3,   0,   0],
 [  0,   0,   0,   0,   2,   2,   0,   0,   0,   0],
 [  0,   4,   4,   4,   4,   4,   4,   4,   4,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0, 104, 0, 0,   0,   0, 102, 0, 0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],

[[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0, 101, 100, 200,   0,   0, 103, 100, 201,   0],
 [  0,   4,   4,   4,   4,   4,   4,   4,   4,   0],
 [  0,   0,   0,   0,   2,   2,   0,   0,   0,   0],
 [  0,   0,   1,   0,   2,   2,   0,   0,   0,   0],
 [  0,   0,   0,   0,   2,   2,   0,   0,   0,   0],
 [  0,   4,   4,   4,   4,   4,   4,   4,   4,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0, 104, 100, 203,   0,   0, 102, 100, 203,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],

[[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],

[[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0, 101, 100, 200,   0,   0,   0,   0,   0,   0],
 [  0,   4,   4,   4,   4,   4,   4,   4,   4,   0],
 [  0,   0,   0,   0,   2,   2,   0,   0,   0,   0],
 [  0,   0,   1,   0,   2,   2,   0,   3,   0,   0],
 [  0,   0,   0,   0,   2,   2,   0,   0,   0,   0],
 [  0,   4,   4,   4,   4,   4,   4,   4,   4,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0, 104, 100, 203,   0,   0, 102, 100, 203,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]]
]

def reward_function(obs):
    text_object = [101,102,103,104,105,106,107]
    score = 0
    obs = np.array(obs)
    # 1. something is you = is it playable?
    score_sub = 0.4
    index_is = np.where(obs == 100)[0]
    index_check = index_is % 10
    index_check = np.where((index_check==0)|(index_check==9))[0] # is가 맨 끝에 있는지 확인
    index_is = np.delete(index_is, index_check)
    
    value_check = obs[index_is+1] # is 오른쪽에 you가 있는지 확인
    value_check = np.where(value_check == 200)

    value_check = obs[index_is[value_check]-1] # is 왼쪽에 object가 있는지 확인
    value_check_list = np.empty((1))
    value_check_list = np.delete(value_check_list, 0)
    for value in value_check:
        if value % 100 != 0 and value // 100 == 1:
            value = value % 100
            found_value = np.where(obs == value)[0]
            value_check_list = np.append(value_check_list, found_value)
    
    if not(value_check_list.any()):
        score_sub = 0

    score = score + score_sub
    
    # 2. is win = is it solvable?
    score_sub = 0.4
    num_is = len(np.where(obs == 100)[0])
    num_win = len(np.where(obs == 201)[0])
    if not(num_is >= 2 and num_win >= 1):
        score_sub = 0

    score = score + score_sub
    
    # 3. 0 is more than 30 = is it have enough space?
    score_sub = 0.1
    num_empty = len(np.where(obs == 0)[0])
    if num_empty < 30:
        score_sub = 0

    score = score + score_sub

    # 4. all 10x text are pared with x objects = is it make sense?
    score_sub = 0.1
    for i in text_object:
        found_object_index = np.where(obs==i)[0]
        if found_object_index.any():
            found_object = np.where(obs==(i%100))[0]
            if not(found_object.any()):
                score_sub = 0
    
    score = score + score_sub
    return score

def reward_cal(before_action, after_action):
    before_score = reward_function(before_action)
    after_score = reward_function(after_action)
    diffenrece_score = after_score - before_score
    return diffenrece_score

def array_to_json():
    return 0

epsilon = 0.9
max_episode = 10000
memory_size = 10000
start_training = 256
batch_size = 32
gamma = 0.9
target_update_step = 1000
learning_rate = 0.01

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc0 = nn.Linear(100,256)
        self.fc1 = nn.Linear(256,512)
        self.fc2 = nn.Linear(512,24)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

    def get_action(self, obs):
        obs = to_tensor(obs)
        action = self.forward(obs)
        if epsilon > np.random.uniform():
            action = to_tensor(np.random.rand(24))
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
    np.random.seed(2)
    model = DQN()
    target = copy.deepcopy(model)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    replay_memory = [None] * memory_size
    idx = 0

    for i in range(1,max_episode):
        done = 0
        t_r = 0
        random_seed = np.random.randint(len(pcg_seed))
        obs = to_tensor(pcg_seed[random_seed])
        obs = obs.view(-1)
        obs = obs.tolist()

        for j in range(len(obs)):
            with torch.no_grad():
                action = model.get_action(obs)
                obs_ = copy.deepcopy(obs)
                obs_[j] = action_bag[action]
                r = reward_cal(obs,obs_)
                if j == 99:
                    done = 1

                replay_memory[idx%memory_size] = [obs, action, r, obs_,not done]
                
                epsilon *= 0.99
                idx += 1
                t_r += r
                obs = obs_

            if idx >= start_training:
                if idx % target_update_step == 0:
                    target = copy.deepcopy(model)
                train(model, target, random.sample(replay_memory[:idx], batch_size), optimizer)
        
        if i % 1000 == 0:
            torch.save(model, 'PCGRL'+str(i)+'.h5')
            # 불러오기
            # 모델 클래스는 어딘가에 반드시 선언되어 있어야 합니다
            # model = torch.load(PATH)
            # model.eval()
        print("episode {} total reward : {}".format(i, t_r))

def test():
    model = torch.load('PCGRL10000.h5')
    model.eval()
    random_seed = 4
    
    done = 0
    obs = to_tensor(pcg_seed[random_seed])
    obs = obs.view(-1)
    obs = obs.tolist()

    for j in range(len(obs)):
        with torch.no_grad():
            action = model.get_action(obs)
            obs_ = copy.deepcopy(obs)
            obs_[j] = action_bag[action]
            r = reward_cal(obs,obs_)
            obs = obs_
            if j == 99:
                done = 1
    obs = np.array(obs)
    print(obs.reshape((10,10)))

if __name__ == '__main__':
    main()
    test()