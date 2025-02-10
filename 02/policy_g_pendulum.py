import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import wandb
from tqdm import trange

device = "cpu"
print(f"Using device: {device}")

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean_layer = nn.Linear(128, action_dim)

        self.fc11 = nn.Linear(state_dim, 128) # 与 mean 分支对称
        self.fc12 = nn.Linear(128, 128) # 与 mean 分支对称
        self.logstd_layer = nn.Linear(128, action_dim) # 与 mean 分支对称


        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.mean_layer(x)) * 2

        xx = torch.relu(self.fc11(state)) # 与 x 计算对称
        xx = torch.relu(self.fc12(xx)) # 与 x 计算对称
        logstd = self.logstd_layer(xx)
        return mean, logstd

class PolicyGradientTrainer:
    def __init__(self, env, model, optimizer, gamma=0.99):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.episode_rewards = []

    def compute_returns(self, rewards):
        discounted_rewards = []
        running_return = 0
        for r in reversed(rewards):
            running_return = r + self.gamma * running_return
            discounted_rewards.insert(0, running_return)
        return torch.tensor(discounted_rewards, dtype=torch.float32).to(device)

    def train(self, num_episodes=5000):
        for episode in trange(num_episodes):
            state, _ = self.env.reset()
            state = torch.FloatTensor(state).to(device)
            done = False
            truncated = False
            rewards = []
            log_probs = []

            while not done and not truncated:
                mean, logstd = self.model(state)
                std = torch.exp(logstd) # 移除 / 100
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()

                action_clamped = torch.clamp(action, min=-2, max=2)

                next_state, reward, done, truncated, _ = self.env.step(action_clamped.cpu().numpy())


                rewards.append(reward)
                log_probs.append(log_prob)
                state = torch.FloatTensor(next_state).to(device)

            if len(rewards) == 0:
                continue

            returns = self.compute_returns(rewards)
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)

            log_probs = torch.stack(log_probs)
            loss = -torch.mean(log_probs * returns)

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # 可以尝试调整或暂时移除

            self.optimizer.step()

            total_reward = sum(rewards)
            self.episode_rewards.append(total_reward)
            wandb.log({
                "episode": episode,
                "reward": total_reward,
                "loss": loss.item(),
                "mean_std": std.mean().item() # 监控 std 的均值
            })

            if episode % 100 == 0:
                print(f"Episode {episode}, Reward: {total_reward:.1f}, Loss: {loss.item():.4f}, Mean Std: {std.mean().item():.4f}") # 打印 loss 和 std

def main():
    wandb.init(project="rl")
    env = gym.make("Pendulum-v1")
    model = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4) # 尝试增大 lr

    trainer = PolicyGradientTrainer(env, model, optimizer, gamma=0.99)
    trainer.train()

    torch.save(model.state_dict(), "policy_model.pth")
    test(model)

def test(model):
    env = gym.make("Pendulum-v1", render_mode="human")
    state, _ = env.reset()
    total_reward = 0

    while True:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            mean, logstd = model(state_tensor)
            action = torch.clamp(Normal(mean, torch.exp(logstd)).sample(), min=-2, max=2)
            next_state, reward, done, _, _ = env.step(action.cpu().numpy())
            total_reward += reward
            state = next_state
            if done:
                break
    print(f"Test Reward: {total_reward:.1f}")
    env.close()

if __name__ == "__main__":
    main()