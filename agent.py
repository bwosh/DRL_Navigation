import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from buffer import ReplayBuffer
from model import QNetwork
from utils import soft_update

class Agent():
    def __init__(self, state_size, action_size, seed=0, lr=1e-3, update_every=4, batch_size=4, buffer_size=64, gamma = 0.0994,tau = 1e-3,  model_path="model.pth"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("=== AGENT ===")
        print(f"Created agent on device: {self.device}")

        self.model_path = model_path
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.update_every = update_every
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        # network variables
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.load()

        # Control variables
        self.memory = ReplayBuffer(action_size, buffer_size, self.batch_size, seed, self.device)
        self.t_step = 0

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss and backpropagate
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)  

    def save(self):
        torch.save(self.qnetwork_target.state_dict(),self.model_path)
        torch.save(self.qnetwork_target.state_dict(),self.model_path.replace('.pth','2.pth'))
        print("Saved agent model.")

    def load(self):
        if( os.path.isfile(self.model_path)):
            self.qnetwork_local.load_state_dict(torch.load(self.model_path))
            self.qnetwork_target.load_state_dict(torch.load(self.model_path))
            print(f"Loaded agent model: {self.model_path}")