from envs.param import *
from utils import *

import gym 
from gym.envs.registration import register
import imageio
import numpy as np 
import random 
import os

register(
    id='Shooter-v0',
    entry_point='envs:ShooterEnv',
)

env = gym.make('Shooter-v0').env
env.reset()

ALPHA = 0.1
GAMMA = 0.99
epsilon = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
num_epochs = 5000
# state size = 2 (presence or absence of enemies) ^ 8 (8 directions)
filename = 'plots/qleraning.png'
q_table = np.zeros([256, 8])
frames = []
scores = []

for i in range(num_epochs):
    state = env.reset()
    done = False 
    
    while not done:
        if 0 <= i % 1000 <= 5: # record the agent for few trials every 1000 epochs 
            frame = env.render(mode='rgb_array')
            frames.append(label_with_episode(frame, i))
        
        else:
            env.render()
        state_num = translate_state(state)
        epsilon = max(MIN_EPSILON, epsilon*EPSILON_DECAY)
        if random.uniform(0, 1) < epsilon: # exploration 
            action = env.action_space.sample()
        else: # exploitation 
            action = np.argmax(q_table[state_num]) 

        next_state, reward, done, info = env.step(action) 
        
        old_value = q_table[state_num, action]
        new_state_num = translate_state(next_state)
        next_max = np.max(q_table[new_state_num])
        
        new_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
        q_table[state_num, action] = new_value

        state = next_state
    scores.append(reward)

print("Finish Training")
env.close()

x = [i+1 for i in range(num_epochs)]
plotLearning(x, scores, filename)

# save the loaded frames 
imageio.mimwrite(os.path.join('./videos/', 'qlearing_agent.gif'), frames, fps=60)

# saving the q-table 
np.savetxt("q_agent.csv", q_table, delimiter=',')