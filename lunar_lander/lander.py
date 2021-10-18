"""
This file runs the lander game
"""
import gym
import numpy as np
#import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

# Import personal library
from reinforce import REINFORCE
#from datetime import datetime
#%%
# -----------------------------------------------------------------------------
#                            GUI Development
# -----------------------------------------------------------------------------
app = QtGui.QApplication([])
# Creating window
win = pg.GraphicsWindow(title='Lunar Lander Data')
win.resize(1000, 600)
win.setWindowTitle('RL Data')
# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)
# Adding window for plots
p1 = win.addPlot(title = 'Reward vs Iteration')

curve = p1.plot(pen='y')
## ------------------------------
env = gym.make('LunarLander-v2')
env = env.unwrapped

# Policy gradient has high variance, seed for reproducibility
env.seed(1)

print("env.action_space", env.action_space)
print("env.observation_space", env.observation_space)
print("env.observation_space.high", env.observation_space.high)
print("env.observation_space.low", env.observation_space.low)


EPISODES = 1000
rewards = []

learning_algorithm = REINFORCE(
    num_inputs=env.observation_space.shape[0], num_actions=env.action_space.n,
    learning_rate=0.02, discount_factor=0.95)

reward_history=[]
p1.setRange(xRange=[0, EPISODES+1], yRange=[-400, 400])

for episode in range(EPISODES):
    observation = env.reset()
    episode_reward = 0
    done = False
    steps=0
    while not(done):
        curve.setData(reward_history, symbol='o')
        if episode%50==0: env.render()

        # 1. Choose an action based on observation
        action = learning_algorithm.run_policy(observation)

        # 2. Take action in the environment
        observation_, reward, done, info = env.step(action)
        steps+=1

        # 4. Store transition for training
        learning_algorithm.store_transition(observation, action, reward)

        if steps>400:
            done = True

        episode_rewards_sum = sum(learning_algorithm.episode_rewards)
        if episode_rewards_sum < -250:
            done = True

        if done:
            episode_rewards_sum = sum(learning_algorithm.episode_rewards)
            rewards.append(episode_rewards_sum)
            max_reward_so_far = np.amax(rewards)

            print(40*"=")
            print(f'Episode: {episode}')
            print(f'Steps: {steps}')
            print(f'Reward: {episode_rewards_sum}')
            print(f'Max reward so far: {max_reward_so_far}')
            reward_history.append(episode_rewards_sum)
            # 5. Train neural network
            discounted_episode_rewards_norm = learning_algorithm.learn()


            # if episode%10==0:
            #     curve.setData(reward_history, symbol='o')

        # Save new observation
        observation = observation_
        
#plt.savefig("Result_"+datetime.now().strftime("%m%d%Y_%H%M%S")+".png")
