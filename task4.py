## Importing all the necessary libraries

import pandas as pd
import numpy as np
import torch
import random
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim 
from mini_pong import MiniPongEnv
from matplotlib import pyplot as plt
import time
import timeit
from tqdm import tqdm


############################################## Question 4 - RL(Level 3) ###################################################################

##############################Configuration for Hyperparameters ###############################

Parameters = {
    "gamma": 0.999,             # Increase discount factor to prioritize future rewards
    "epsilon": 1.0,             # Initial exploration rate
    "epsilon_min": 0.1,         # Minimum exploration rate
    "epsilon_decay": 0.995,      # Slower epsilon decay for better early exploration
    "learning_rate": 0.0001,    # Reduced learning rate for stable updates
    "hidden_dim": 128,           # Size of hidden layer in the neural network
    "episodes": 5000,           # Increased number of training episodes
}


##############################Creating model ###############################

# Define the DQNN
class DQNN(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, lr):
        super(DQNN, self).__init__()
        
        # First fully connected layer maps input to hidden dimension
        self.fcl1 = nn.Linear(in_dim, hid_dim)
        
        # Second fully connected layer maps hidden dimension to twice its size
        self.fcl2 = nn.Linear(hid_dim, hid_dim * 2)
        
        # Output layer maps to the output dimension
        self.fcl3 = nn.Linear(hid_dim * 2, out_dim)

        # Loss function for Q-value prediction
        self.criterion = nn.MSELoss()
        
        # Optimizer with specified learning rate
        self.optimizer = optim.Adam(self.parameters(), lr)

    # Forward pass
    def forward(self, x):
        # Pass input through the first layer with ReLU activation
        x = torch.relu(self.fcl1(x))
        
        # Pass result through the second layer with ReLU activation
        x = torch.relu(self.fcl2(x))
        
        # Pass result through the output layer without activation
        return self.fcl3(x)
    
    # Update function to train the model on a single pair
    def update(self, input, target):
        # Perform forward pass to get the prediction
        prediction = self(input)
        
        # Calculate the loss between prediction and target
        loss = self.criterion(prediction, target)
        
        # Reset gradients to prepare for backpropagation
        self.optimizer.zero_grad()
        
        # Perform backpropagation to compute gradients
        loss.backward()
        
        # Update model parameters based on gradients
        self.optimizer.step()

    # Predict function to return Q-values without computing gradients
    def predict(self, input):
        # Disable gradient calculations during prediction
        with torch.no_grad():
            return self(input)

############################## Training model ###############################

def q_learning(env, model, episodes, gamma, epsilon, epsilon_decay, epsilon_min):
    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice([0, 1, 2])  # Action space: [left, stay, right]
            else:
                q_values = model.predict(torch.tensor(state, dtype=torch.float32))
                action = torch.argmax(q_values).item()

            # Take action and observe outcome
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Get Q-values and calculate the target
            q_values = model.predict(torch.tensor(state, dtype=torch.float32)).clone().detach()
            if done:
                q_values[action] = reward  # End of episode update
            else:
                q_values_next = model.predict(torch.tensor(next_state, dtype=torch.float32))
                q_values[action] = reward + gamma * torch.max(q_values_next).item()

            # Update the model with the new Q-values
            model.update(torch.tensor(state, dtype=torch.float32), q_values)

            # Move to the next state
            state = next_state

        # Decay epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Record total reward for this episode
        rewards_per_episode.append(total_reward)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Total Reward = {total_reward}, Epsilon = {epsilon}")

    # Plot training rewards
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards over Episodes")
    plt.show()

    return rewards_per_episode

# Initialize DQN model
env = MiniPongEnv(level=3, size=5, normalise=True)
n_state = 3         
n_action = 3

model = DQNN(in_dim=n_state, out_dim=n_action, hid_dim = Parameters["hidden_dim"], lr = Parameters["learning_rate"])

# Train the model
training_rewards = q_learning(env=env, model = model, episodes = Parameters["episodes"], 
                              gamma = Parameters["gamma"], epsilon = Parameters["epsilon"], 
                              epsilon_decay = Parameters["epsilon_decay"],epsilon_min = Parameters["epsilon_min"])


############################## Testing model ###############################

def test_qlearning(env, model, episodes = 50):
    total_test_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        test_rewards_per_episode = 0

        while not done:
            # Select the best action (no exploration)
            test_q_values = model.predict(torch.tensor(state, dtype=torch.float32))
            test_action = torch.argmax(test_q_values).item()

            # Take action in environment
            next_state, test_reward, done, _ = env.step(test_action)
            test_rewards_per_episode += test_reward
            state = next_state

        total_test_rewards.append(test_rewards_per_episode)

        # Calculate average and standard deviation of rewards
    test_average_reward = np.mean(total_test_rewards)
    test_reward_std_dev = np.std(total_test_rewards)

    print(f"Test Average Reward: {test_average_reward}")
    print(f"Test Reward Standard Deviation: {test_reward_std_dev}")


# Plotting test rewards
    plt.plot(total_test_rewards, label='Rewards per Episode')
    plt.axhline(y=test_average_reward, color='r', linestyle='--', label=f'Average Reward ({test_average_reward:.2f})')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.title("Test Rewards per Episode")
    plt.show()

    return test_average_reward, test_reward_std_dev    
    

# Set epsilon to 0 for pure exploitation
test_average, test_std_dev = test_qlearning(env, model, episodes=50)







