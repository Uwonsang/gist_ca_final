from baba_is_gym import Env, time_measure, get_keyboard_input
import cv2
import numpy as np

def pre_processing(state):
    state = np.dot(state[...,:3], [0.2989, 0.5870, 0.1140])
    state = cv2.resize(state,(84, 84), interpolation=cv2.INTER_NEAREST)
    state = state/state.max()
    state = state.reshape((1,1,84,84))
    return state

import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(1,8,(8, 8),stride=4,padding=2)
        self.conv2 = nn.Conv2d(8,16,(4, 4),stride=3,padding=1)
        self.conv3 = nn.Conv2d(16,16,(3, 3),stride=1,padding=1)
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.flatten(x,1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,4)
        else : 
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    s,a,r,s_prime,done_mask = memory.sample(batch_size)

    s = s.reshape((batch_size,1,84,84))
    s_prime = s_prime.reshape((batch_size,1,84,84))

    q_out = q(s)
    q_a = q_out.gather(1,a)
    max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
    target = r + gamma * max_q_prime * done_mask
    loss = F.smooth_l1_loss(q_a, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

import pickle

def main():
    t = time_measure()
    r = 0
    for j in range(8):
        env = Env(j+3,True) # 2 ~ 9
        cuda = torch.device('cuda')
        torch.cuda.empty_cache()
        q = Qnet()
        q_target = Qnet()

        q_target.load_state_dict(q.state_dict())
        memory = ReplayBuffer()

        print_interval = 20
        optimizer = optim.Adam(q.parameters(), lr=learning_rate)
        
        print('training stage',j+1)
        score_per_episode = []
        for n_epi in range(500):
            score = 0.0
            epsilon = max(0.01, 0.08 - 0.01*(n_epi/50)) #Linear annealing from 8% to 1%
            s, _ = env.reset()
            s = pre_processing(s[0])
            done = False
            if n_epi % 100 == 0:
                print('pass',n_epi,'episodes')
            for i in range(200):
                t.start()
                a = q.sample_action(torch.from_numpy(s).float(), epsilon)
                s_prime, r, done, info = env.step(a)
                s_prime = pre_processing(s_prime[0])
                if done == 0 and i == 199:
                    r = -10
                done_mask = 0.0 if done else 1.0
                memory.put((s[0],a,r/100.0,s_prime[0], done_mask))
                s = s_prime
                r += t.end()
                score += r
                if done:
                    break
                
                if memory.size()>200:
                    train(q, q_target, memory, optimizer)

            if n_epi % 50:
                q_target.load_state_dict(q.state_dict())
            score_per_episode.append(score)
        with open('stage'+str(j+1)+'_DQN.pkl', 'wb') as f:
            pickle.dump(score_per_episode, f)

            # if n_epi%print_interval==0 and n_epi!=0:
            #     q_target.load_state_dict(q.state_dict())
            #     print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
            #                                                     n_epi, score/print_interval, memory.size(), epsilon*100))
            #     score = 0.0

if __name__ == '__main__':
    # for j in range(8):
    #     with open('stage'+str(j+1)+'_DQN.pkl', 'rb') as f:
    #             test = pickle.load(f)
    #             print(np.asarray(test).mean())
    main()