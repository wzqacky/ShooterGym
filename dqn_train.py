import gym
from gym.envs.registration import register
import imageio
import os 
import torch 
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import Agent
from utils import *

if __name__ == '__main__':
    register(
        id='Shooter-v0',
        entry_point='envs:ShooterEnv',
    )

    env = gym.make('Shooter-v0').env
    env.reset()
    num_epochs = 5000 
    load_checkpoint = False

    agent = Agent(gamma=0.99, epsilon=1.0, lr=1e-3,
                  input_dims=19, n_actions=8, eps_end=0.01,
                  batch_size=64)

    if load_checkpoint:
        agent.load_models()

    filename = 'plots/dqn_result.png'
    scores = []
    eps_history = []
    n_steps = 0
    frames = []

    for i in range(num_epochs):
        done = False
        observation = env.reset()
        observation = observation.astype(np.float32)

        while not done:
            if 0 <= i % 1000 <= 5: # record the agent for few trials every 1000 epochs 
                frame = env.render(mode='rgb_array')
                frames.append(label_with_episode(frame, i))
            else:
                env.render()

            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            observation = observation.astype(np.float32)
            observation_ = observation_.astype(np.float32)
            agent.store_transition(observation, action,
                                    reward, observation_, int(done))
            agent.learn()

            observation = observation_

        scores.append(reward)
        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('episode: ', i,'score %.1f ' % reward,
             ' average score %.1f' % avg_score,
            'epsilon %.2f' % agent.epsilon)
        #if i > 0 and i % 10 == 0:
        #    agent.save_models()

        eps_history.append(agent.epsilon)

    env.close()

    # save the loaded frames 
    imageio.mimwrite(os.path.join('./videos/', 'dqn_agent.gif'), frames, fps=60)

    # plotting the graph 
    x = [i+1 for i in range(num_epochs)]
    plotLearning(x, scores, filename)

    # saving the agent 
    agent.save_model()