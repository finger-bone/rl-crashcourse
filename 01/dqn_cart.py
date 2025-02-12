import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import trange
import os

wandb.init(project="rl")
device = "cpu"
print(f"Using device: {device}")

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  
        self.fc2 = nn.Linear(128, 128) 
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x) + x)
        x = self.fc3(x)
        return x

num_episodes = 5000
learning_rate = 3e-4
gamma = 0.99
epsilon = 1.0  
epsilon_decay = 0.99 
min_epsilon = 0.005

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n 

q_network = QNetwork(state_dim, action_dim).to(device)
if os.path.exists("q_network.pth"):
    q_network.load_state_dict(torch.load("q_network.pth"))
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, num_episodes)
loss_fn = nn.MSELoss()

for episode in trange(num_episodes):
    state, _ = env.reset() 
    state = torch.FloatTensor(state).to(device) 
    episode_reward = 0 
    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_network(state)
            action = torch.argmax(q_values).item()
        next_state, reward, done, _, _ = env.step(action)
        next_state_tensor = torch.FloatTensor(next_state).to(device) 
        episode_reward += reward

        with torch.no_grad():
            q_next = q_network(next_state_tensor)
            max_q_next = torch.max(q_next).item()
            target_q = reward + (gamma * max_q_next if not done else 0.0)
        
        q_values = q_network(state)
        current_q = q_values[action]
        
        loss = loss_fn(current_q, torch.tensor(target_q).to(device)) 
        optimizer.zero_grad()
        loss.backward()
        wandb.log({"loss": loss.item(), "episode": episode, "lr": scheduler.get_last_lr()[0], "epsilon": epsilon})
        optimizer.step()
        state = next_state_tensor
    wandb.log({"reward": episode_reward})
    scheduler.step()
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

env.close()

q_network.eval()
torch.save(q_network.state_dict(), "q_network.pth")
test_env = gym.make("CartPole-v1", render_mode="human")
state, _ = test_env.reset()
done = False

while not done:
    test_env.render()
    state_tensor = torch.FloatTensor(state).to(device)
    with torch.no_grad():
        q_values = q_network(state_tensor)
    action = torch.argmax(q_values).item()
    state, reward, done, _, _ = test_env.step(action)

test_env.close()