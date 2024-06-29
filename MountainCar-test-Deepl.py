import numpy as np
import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

env = gym.make("MountainCar-v0", render_mode='human')

class Network(nn.Module):
    
    def __init__(self, state_size, action_size):
        super(Network,self).__init__()
        
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,action_size)
        
    def forward(self, state):
        x =  self.fc1(state)
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
        
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences ]).astype(np.uint8)).float()
        
        return (states, actions, rewards, next_states, dones)

REPLAY_BUFFER_SIZE = 100_000
learning_rate = 1e-4
discount_factor = 0.99
interpolation_parameter = 1e-3
  
class Agent():
    
    def __init__(self, state_size, action_size):
        
        self.local_network = Network(state_size, action_size)
        self.target_network = Network(state_size, action_size)
        self.memory = Replay_Memory(REPLAY_BUFFER_SIZE)
        self.optimizer = optim.Adam(self.local_network.parameters(), lr=learning_rate)
        self.t_step = 0
    
    def act(self, state, epsilon = 0.):
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        self.local_network.eval()
        with torch.no_grad():
            action_values = self.local_network(state)
        self.local_network.train()
        if np.random.random() > epsilon:
            return np.argmax(action_values.data.numpy())
        else:
            return random.choice(np.arange(3))
    
    def step(self, state, action, reward, next_state, done):
        self.memory.push((state,action,reward,next_state,done))
        self.t_step = (self.t_step + 1) % 4
        
        if self.t_step == 0:
            if len(self.memory.memory) > 100:
                experiences = self.memory.sample(100)
                self.learn(experiences, discount_factor)
                
    def learn(self, experiences, discount_factor):
        states, actions, rewards, next_states, dones = experiences
        
        Q_Targets_Next = self.target_network(next_states).float().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (discount_factor * Q_Targets_Next * (1 - dones))
        
        Q_expected = self.local_network(states).gather(1,actions)
        
        loss = F.mse_loss(Q_expected,Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_network, self.target_network, interpolation_parameter)
    
    def soft_update(self, local_network, target_network, interpolation_parameter):
        for target_network, local_network in zip(target_network.parameters(), local_network.parameters()):
            target_network.data.copy_(interpolation_parameter * local_network.data + (1 - interpolation_parameter) * target_network.data)

def save_model(agent, filename):
    checkpoint = {
        'local_network_state_dict': agent.local_network.state_dict(),
        'target_network_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'memory': agent.memory.memory,
    }
    torch.save(checkpoint, filename)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = Agent(state_size=state_size, action_size=action_size)            

def load_model(agent, filename):
    checkpoint = torch.load(filename)
    agent.local_network.load_state_dict(checkpoint['local_network_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.memory.memory = checkpoint['memory']

load_model(agent=agent,filename=r"mountaincar_dqn.pth")

state,_ = env.reset()
done = False

while not done:
    action = agent.act(state)
    state, reward,done,_,_ = env.step(action)
    env.render()
env.close()
