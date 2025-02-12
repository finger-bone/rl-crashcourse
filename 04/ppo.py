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

class PPOTrainer:
    def __init__(self, env, actor_model, critic_model, actor_optimizer, critic_optimizer,
                 gamma=0.99, gae_lambda=0.95, clip_param=0.2, n_epochs=4, batch_size=64):
        self.env = env
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.episode_rewards = []

    def compute_gae(self, rewards, values, dones, next_value):
        values = values + [next_value]
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages, dtype=torch.float32).to(device)

    def train(self, num_updates=1000, n_steps=2048):
        for update in trange(num_updates):
            states, actions, rewards, dones, old_log_probs, values = [], [], [], [], [], []
            state, _ = self.env.reset()
            state = torch.FloatTensor(state).to(device)
            episode_reward = 0
            
            for _ in range(n_steps):
                with torch.no_grad():
                    mean, logstd = self.actor_model(state)
                    std = torch.exp(logstd)
                    dist = Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum()
                    value = self.critic_model(state).squeeze()
                    
                next_state, reward, done, truncated, _ = self.env.step(action.cpu().numpy())
                episode_reward += reward
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done or truncated)
                old_log_probs.append(log_prob)
                values.append(value)
                
                state = torch.FloatTensor(next_state).to(device) if not (done or truncated) else torch.FloatTensor(self.env.reset()[0]).to(device)
                
                if done or truncated:
                    self.episode_rewards.append(episode_reward)
                    episode_reward = 0

            with torch.no_grad():
                next_value = self.critic_model(state).squeeze().item()
            
            states = torch.stack(states)
            actions = torch.stack(actions)
            old_log_probs = torch.stack(old_log_probs)
            values = torch.stack(values).cpu().numpy()
            
            advantages = self.compute_gae(rewards, values.tolist(), dones, next_value)
            returns = advantages + torch.tensor(values, dtype=torch.float32).to(device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advantages)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            for _ in range(self.n_epochs):
                for batch in dataloader:
                    b_states, b_actions, b_old_log_probs, b_returns, b_advantages = batch
                    
                    mean, logstd = self.actor_model(b_states)
                    std = torch.exp(logstd)
                    dist = Normal(mean, std)
                    new_log_probs = dist.log_prob(b_actions).sum(dim=1)
                    entropy = dist.entropy().mean()
                    
                    ratio = torch.exp(new_log_probs - b_old_log_probs)
                    surr1 = ratio * b_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * b_advantages
                    actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
                    
                    current_values = self.critic_model(b_states).squeeze()
                    critic_loss = nn.MSELoss()(current_values, b_returns)
                    
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), 0.5)
                    self.actor_optimizer.step()
                    
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), 0.5)
                    self.critic_optimizer.step()
            
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                wandb.log({
                    "update": update,
                    "avg_reward": avg_reward,
                    "actor_loss": actor_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "entropy": entropy.item()
                })
                if update % 10 == 0:
                    print(f"Update {update}, Avg Reward: {avg_reward:.1f}")

def main():
    wandb.init(project="rl")
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor_model = ActorNetwork(state_dim, action_dim).to(device)
    critic_model = CriticNetwork(state_dim).to(device)

    actor_optimizer = optim.Adam(actor_model.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic_model.parameters(), lr=3e-4)

    trainer = PPOTrainer(
        env=env,
        actor_model=actor_model,
        critic_model=critic_model,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        n_epochs=4,
        batch_size=64
    )
    trainer.train(num_updates=500, n_steps=2048)

    torch.save(actor_model.state_dict(), "ppo_actor.pth")
    torch.save(critic_model.state_dict(), "ppo_critic.pth")
    test(actor_model)

def test(actor_model):
    env = gym.make("Pendulum-v1", render_mode="human")
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