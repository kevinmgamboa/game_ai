# An Implementation of REINFORCE
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class REINFORCE:
    def __init__(self,num_inputs,num_actions,learning_rate=0.01,discount_factor=0.95):
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.optimizer=keras.optimizers.Adam(self.lr)        
        self.num_actions=num_actions
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

        # Build the neural-network stochastic policy network:
        hidden_nodes=[6,6]
        inputs = keras.Input(shape=(num_inputs,))
        x=inputs
        x_ = layers.Dense(hidden_nodes[0], activation="tanh")(x)
        x=tf.concat([x,x_],axis=1) # This passes shortcut connections from all earlier layers to this next one
        x_ = layers.Dense(hidden_nodes[1], activation="tanh")(x)
        x=tf.concat([x,x_],axis=1) # This passes shortcut connections from all earlier layers to this next one
        outputs = layers.Dense(num_actions, activation="softmax")(x)
        self.policy_network = keras.Model(inputs=inputs, outputs=outputs, name="policy_network")


    def store_transition(self, s, a, r):
        """
            Store play memory for training

            Arguments:
                s: observation
                a: action taken
                r: reward after action
        """
        self.episode_observations.append(s)
        self.episode_rewards.append(r)
        # Store actions as list of one-hot-encoded actions
        action = np.zeros(self.num_actions)
        action[a] = 1
        self.episode_actions.append(action)


    def run_policy(self, observation):
        # Reshape observation to (1,num_features)
        observation = observation[np.newaxis,:]
        # Run forward propagation to get softmax probabilities
        prob_weights = self.policy_network(observation).numpy()
        # Select action using a biased sample
        # this will return the index of the action we've sampled
        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())
        return action

    def learn(self):
        # Discount and normalize episode reward
        discounted_episode_rewards_norm = self.discount_and_norm_rewards()

        # Train on episode
        
        batch_trajectory=np.vstack(self.episode_observations)
        batch_action_choices=np.argmax(np.vstack(np.array(self.episode_actions)).astype(np.int32),axis=1)
        print("batch_trajectory",batch_trajectory.shape)
        print("batch_action_choices",batch_action_choices.shape)
        print("discounted_episode_rewards_norm",discounted_episode_rewards_norm.shape)
        with tf.GradientTape() as tape:
            trajectory_action_probabilities=self.policy_network(batch_trajectory)
            chosen_probabilities=tf.gather(trajectory_action_probabilities,indices=batch_action_choices,axis=1, batch_dims=1) # this returns a tensor of shape [trajectory_length]
            log_probabilities=tf.math.log(chosen_probabilities)
            logprobrewards=log_probabilities*discounted_episode_rewards_norm[:]
            L=-tf.reduce_sum(logprobrewards) # This minus sign here is because the apply_gradients only does gradient DESCENT, but we want ASCENT!
        grads = tape.gradient(L, self.policy_network.trainable_weights) # This calculates the gradient required by REINFORCE
        self.optimizer.apply_gradients(zip(grads, self.policy_network.trainable_weights)) # This updates the parameter vector '''

        # Reset the episode data
        self.episode_observations, self.episode_actions, self.episode_rewards  = [], [], []

        return discounted_episode_rewards_norm


    def discount_and_norm_rewards(self):
        discounted_episode_rewards = np.zeros_like(self.episode_rewards)
        cumulative = 0
        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.discount_factor + self.episode_rewards[t]
            discounted_episode_rewards[t] = cumulative

        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards 
    
