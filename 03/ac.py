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

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean_layer = nn.Linear(128, action_dim)
        self.logstd_layer = nn.Linear(128, action_dim)

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean_layer(x)
        logstd = self.logstd_layer(x)
        logstd = torch.clamp(logstd, min=-20, max=2)
        return mean, logstd

class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value_layer = nn.Linear(128, 1)

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.value_layer(x)
        return value


class ActorCriticTrainer:
    def __init__(self, env, actor_model, critic_model, actor_optimizer, critic_optimizer, gamma=0.99):
        self.env = env
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
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
        epsilon = 0.9
        epsilon_decay = 0.99
        min_epsilon = 0.001
        for episode in trange(num_episodes):
            state, _ = self.env.reset()
            state = torch.FloatTensor(state).to(device)
            done = False
            truncated = False
            rewards = []
            log_probs = []
            values = []

            while not done and not truncated:
                mean, logstd = self.actor_model(state)
                std = torch.exp(logstd)
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()
                action_clamped = torch.clamp(action, min=-2, max=2)
                if np.random.random() < epsilon:
                    action = self.env.action_space.sample()
                value = self.critic_model(state)

                next_state, reward, done, truncated, _ = self.env.step(action_clamped.cpu().numpy())

                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                state = torch.FloatTensor(next_state).to(device)

            if len(rewards) == 0:
                continue

            returns = self.compute_returns(rewards)
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)

            log_probs = torch.stack(log_probs)
            values = torch.stack(values).squeeze(-1)

            advantages = returns - values.detach()

            critic_loss = nn.MSELoss()(values, returns)

            actor_loss = -1e1 * torch.mean(log_probs * advantages)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), 1.0)
            self.critic_optimizer.step()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), 1.0)
            self.actor_optimizer.step()

            total_reward = sum(rewards)
            self.episode_rewards.append(total_reward)
            wandb.log({
                "episode": episode,
                "reward": total_reward,
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "mean_std": std.mean().item()
            })

            if episode % 100 == 0:
                print(f"Episode {episode}, Reward: {total_reward:.1f}")
            epsilon = max(epsilon * epsilon_decay, min_epsilon)

def main():
    wandb.init(project="rl")
    env = gym.make("Pendulum-v1", g=0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor_model = ActorNetwork(state_dim, action_dim).to(device)
    critic_model = CriticNetwork(state_dim).to(device)

    actor_optimizer = optim.Adam(actor_model.parameters(), lr=8e-5)
    critic_optimizer = optim.Adam(critic_model.parameters(), lr=8e-5)

    trainer = ActorCriticTrainer(env, actor_model, critic_model, actor_optimizer, critic_optimizer, gamma=0.99)
    trainer.train()

    torch.save(actor_model.state_dict(), "actor_model_a2c.pth")
    torch.save(critic_model.state_dict(), "critic_model_a2c.pth")
    test(actor_model)

def test(actor_model):
    env = gym.make("Pendulum-v1", render_mode="human", g=0)
    state, _ = env.reset()
    total_reward = 0

    while True:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            mean, logstd = actor_model(state_tensor)
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