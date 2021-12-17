from baba_is_gym import Env, get_keyboard_input
import numpy as np
from util import time_measure
# w : up
# s : down
# a : left
# d : right

action = 0
env = Env(2, tile_size=40, training_on_single_stage=True)
for j in range(20):
    observation, _ = env.reset()
    done = 0
    i = 0
    while not done:
        action = get_keyboard_input()
        if action != 0:
            i += 1
        env.render()
        observation, reward, done, _ = env.step(action)
        if done:
            env.render()
            break
    print(i)