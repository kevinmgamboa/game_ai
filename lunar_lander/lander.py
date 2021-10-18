import gym
from reinforce import REINFORCE
import matplotlib.pyplot as plt
import numpy as np 
#from datetime import datetime

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
plt.ion()
fig=plt.figure("Reward vs Iteration")
ax=fig.add_subplot(111)
ax.axis([0,EPISODES+1,-400,400])
plt.ylabel('Reward')
plt.xlabel('Iteration')
ax.plot(reward_history, 'b-')
ax.set_title("Learning Rate:"+str(learning_algorithm.lr)+" Discount Factor:"+str(learning_algorithm.discount_factor))
plt.draw()
plt.pause(0.001)

for episode in range(EPISODES):
    observation = env.reset()
    episode_reward = 0
    done = False
    steps=0
    while not(done):
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
            if episode%10==0:
                ax.plot(reward_history, 'b-')
                plt.draw()
                plt.pause(0.001)

        # Save new observation
        observation = observation_
        
#plt.savefig("Result_"+datetime.now().strftime("%m%d%Y_%H%M%S")+".png")
