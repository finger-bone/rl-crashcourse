import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import wandb
from tqdm import trange

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)

class GRPOTrainer:
    def __init__(self, env, actor, optimizer, 
                 clip_ratio=0.2, beta=0.001, gamma=0.99,
                 epochs=10, batch_size=32, 
                 group_size=200):
        self.env = env
        self.actor = actor
        self.optimizer = optimizer
        self.clip_ratio = clip_ratio
        self.beta = beta
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = batch_size
        self.group_size = group_size
        self.ep_rewards = []

    def _calc_returns(self, rewards, dones):
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).to(device)

    def collect_rollout(self):
        states, acts, rews, dones = [], [], [], []
        old_logits = []
        state, _ = self.env.reset()
        state = torch.FloatTensor(state).to(device)
        ep_rew = 0

        for _ in range(self.group_size):
            with torch.no_grad():
                logits = self.actor(state)
                dist = Categorical(logits=logits)
                act = dist.sample()

            next_state, rew, terminated, truncated, _ = self.env.step(act.item())
            done = terminated or truncated
            ep_rew += rew

            states.append(state)
            acts.append(act)
            rews.append(rew)
            dones.append(done)
            old_logits.append(logits)

            state = torch.FloatTensor(next_state).to(device) if not done else torch.FloatTensor(self.env.reset()[0]).to(device)
            
            if done:
                self.ep_rewards.append(ep_rew)
                ep_rew = 0

        returns = self._calc_returns(rews, dones)
        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return (
            torch.stack(states),
            torch.stack(acts),
            advantages,
            torch.stack(old_logits)
        )

    def train(self, total_updates=300):
        self.actor.train()
        for update in trange(total_updates):
            states, actions, advantages, old_logits = self.collect_rollout()
            
            dataset = torch.utils.data.TensorDataset(states, actions, advantages, old_logits)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            policy_losses = []
            kl_divergences = []
            
            for _ in range(self.epochs):
                for batch in loader:
                    s_batch, a_batch, adv_batch, old_logits_batch = batch
                    
                    new_logits = self.actor(s_batch)
                    old_dist = Categorical(logits=old_logits_batch.detach())
                    new_dist = Categorical(logits=new_logits)
                    
                    logp_new = new_dist.log_prob(a_batch)
                    logp_old = old_dist.log_prob(a_batch).detach()
                    ratio = torch.exp(logp_new - logp_old)
                    

                    surr1 = ratio * adv_batch
                    surr2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv_batch
                    policy_loss = -torch.min(surr1, surr2).mean()
                    

                    kl = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()
                    
                    loss = policy_loss + self.beta * kl
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.optimizer.step()
                    
                    policy_losses.append(policy_loss.item())
                    kl_divergences.append(kl.item())
            

            if self.ep_rewards:
                avg_rew = np.mean(self.ep_rewards[-20:])
                wandb.log({
                    "update": update,
                    "avg_reward": avg_rew,
                    "policy_loss": np.mean(policy_losses),
                    "kl_divergence": np.mean(kl_divergences)
                })

def test(env, actor, episodes=5, render=False):
    actor.eval()
    for ep in range(episodes):
        state, _ = env.reset()
        total_rew = 0
        while True:
            if render:
                env.render()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                logits = actor(state_tensor)
                act = torch.argmax(logits).item()
            
            state, rew, terminated, truncated, _ = env.step(act)
            total_rew += rew
            
            if terminated or truncated:
                print(f"Test Episode {ep+1} | Reward: {total_rew}")
                break

def main():
    wandb.init(project="grpo-cartpole")
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    actor = ActorNetwork(state_dim, action_dim).to(device)
    optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    
    trainer = GRPOTrainer(
        env=env,
        actor=actor,
        optimizer=optimizer,
        clip_ratio=0.2,
        beta=0.001,
        gamma=0.99,
        epochs=10,
        batch_size=32,
        group_size=200
    )
    
    trainer.train(total_updates=1000)
    
    test_env = gym.make('CartPole-v1', render_mode='human')
    test(test_env, actor, episodes=3, render=True)
    env.close()

if __name__ == "__main__":
    main()