import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import gymnasium as gym
from torch.autograd import Variable
from collections import deque, namedtuple

env = gym.make("MountainCar-v0")

class Network(nn.Module):
    
    def __init__(self, state_size, action_size):
        super(Network, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        
        return self.fc3(x)
    

class Replay_Memory():
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, experience):
        self.memory.append(experience)
        
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)

        states = torch.from_numpy(np.vstack([experience[0] for experience in experiences])).float()
        actions = torch.from_numpy(np.vstack([experience[1] for experience in experiences])).long()
        rewards = torch.from_numpy(np.vstack([experience[2] for experience in experiences])).float()
        next_states = torch.from_numpy(np.vstack([experience[3] for experience in experiences])).float()
        dones = torch.from_numpy(np.vstack([experience[4] for experience in experiences]).astype(np.uint8)).float()
        
        return (states,actions,rewards,next_states,dones)

REPLAY_BUFFER_SIZE = 100_000
learning_rate = 5e-4
discount_factor = 0.99
interpolation_parameter = 1e-3

class Agent():
    
    def __init__(self, state_size, action_size):
        self.local_network = Network(state_size, action_size)
        self.target_network = Network(state_size, action_size)
        self.memory = Replay_Memory(REPLAY_BUFFER_SIZE)
        self.action_size = action_size
        self.optimizer = optim.Adam(self.local_network.parameters(), lr=learning_rate)
        self.t_step = 0
            
    def step(self, state, action, reward, next_state, done):
        self.memory.push((state,action,reward,next_state,done))
        
        self.t_step = (self.t_step + 1) % 4
        
        if self.t_step == 0:
            if len(self.memory.memory) > 100:
                experiences = self.memory.sample(100)
                self.learn(experiences, discount_factor)
                
    
    def act(self, state, epsilon = 0.):
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        self.local_network.eval()
        with torch.no_grad():
            action_values = self.local_network(state)
        self.local_network.train()
        
        if np.random.random() > epsilon:
            return np.argmax(action_values.data.numpy())
        else:
            return np.random.choice(np.arange(self.action_size))
        
    def learn(self, experiences, discount_factor):
        states, actions, rewards, next_states, dones = experiences
        
        Q_Target_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        Q_Target = rewards + (discount_factor * Q_Target_next * (1 - dones))
        
        Q_Expected = self.local_network(states).gather(1, actions)
        
        loss = F.mse_loss(Q_Expected, Q_Target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.target_network, self.local_network, interpolation_parameter)
        
    def soft_update(self, target_network, local_network, interpolation_parameter = 0.001):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(interpolation_parameter * local_param.data + (1 - interpolation_parameter) * target_param.data)

state_size = env.observation_space.shape[0]
number_actions = env.action_space.n


agent = Agent(state_size, number_actions)

episodes = 2000
epsilon = 1.0
max_number_of_timesteps = 1000
epsilon_decay_value = 0.995

for episode in range(episodes):
    state, _ = env.reset()
    score = 0
    done = False
    
    for t in range(max_number_of_timesteps):
        action = agent.act(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        
        if done:
            break
    
    epsilon = max(epsilon * epsilon_decay_value, 0.01)
    print(f"Total score on episode {episode}: {score}")
            
            
            
            
            
            
            
            
            
            
            
            
            