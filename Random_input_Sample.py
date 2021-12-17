from baba_is_gym import Env, get_keyboard_input
import numpy as np
from util import time_measure
# w : up
# s : down
# a : left
# d : right

action = 0
env = Env(2, tile_size=40, training_on_single_stage=True)
t = time_measure()
r = 0
for j in range(20):
    observation, _ = env.reset()
    done = 0
    t.start()
    for i in range(200):
        action = env.get_random_action()
        env.render()
        observation, reward, done, _ = env.step(action)
        if done:
            break
    r += t.end()
    print('reward :', reward,'time spend :', t.end(), i)
print(r/20)
