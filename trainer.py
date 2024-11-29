import torch
import wandb
import logging
import numpy as np
from pathlib import Path
from typing import Optional
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class DQNTrainer:
    def __init__(
        self,
        env,
        model,
        target_model,
        buffer,
        device: torch.device,
        config
    ):
        self.env = env
        self.device = device
        self.model = model.to(device)
        self.target_model = target_model.to(device)
        self.buffer = buffer
        self.config = config
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.epsilon_final = config.epsilon_final
        self.epsilon_decay = config.epsilon_decay
        self.target_update_freq = config.target_update
        self.total_timesteps = 0
        self.learns = 0
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        if config.use_wandb:
            self.init_wandb()
            wandb.watch(self.model, log_freq=100)
        
        self.update_target_network()
        logger.info("Initialized DQNTrainer with WandB logging")

    def init_wandb(self):
        """Initialize WandB with config parameters"""
        config_dict = {
            "architecture": "DuelingDQN" if self.config.dueling_dqn else "DQN",
            "double_dqn": self.config.double_dqn,
            "learning_rate": self.config.learning_rate,
            "gamma": self.config.gamma,
            "buffer_size": self.config.buffer_size,
            "batch_size": self.config.batch_size,
            "frame_stack": self.config.frame_stack,
            "hidden_size": self.config.hidden_size,
            "epsilon_start": self.config.epsilon_start,
            "epsilon_final": self.config.epsilon_final,
            "epsilon_decay": self.config.epsilon_decay,
            "target_update": self.config.target_update,
            "device": str(self.device)
        }
        wandb.config.update(config_dict)

    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.env.action_space)
        
        with torch.no_grad():
            q_values = self.model(state)
            return q_values.max(1)[1].item()
    
    def update_epsilon(self) -> None:
        if self.epsilon > self.epsilon_final:
            self.epsilon -= self.epsilon_decay

    def train_step(self, batch_size: int) -> Optional[float]:
        if len(self.buffer) < self.config.starting_mem_len:
            return None
            
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            if self.config.double_dqn:
                next_actions = self.model(next_states).max(1)[1]
                next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1))
            else:
                next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
                
            expected_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        if self.config.use_wandb:
            wandb.log({
                "train/loss": loss.item(),
                "train/q_value_mean": current_q_values.mean().item(),
                "train/q_value_max": current_q_values.max().item(),
                "train/q_value_min": current_q_values.min().item(),
                "train/grad_norm": torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf')).item(),
                "train/buffer_size": len(self.buffer),
                "train/epsilon": self.epsilon,
                "train/learning_steps": self.learns
            }, step=self.total_timesteps)
        
        self.learns += 1
        if self.learns % self.target_update_freq == 0:
            self.update_target_network()
            logger.info("Target network updated")
            
        return loss.item()

    def update_target_network(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())
        
    def train_episode(self) -> dict:
        state = self.env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        steps = 0
        
        while not done:
            action = self.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.total_timesteps += 1
            steps += 1
            
            self.buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            if len(self.buffer) > self.config.starting_mem_len:
                loss = self.train_step(self.config.batch_size)
                if loss is not None:
                    episode_loss.append(loss)
                self.update_epsilon()
            
            # Save weights periodically
            if self.total_timesteps % 50000 == 0:
                self.save_checkpoint(self.config.checkpoint_dir / 'recent_weights.pth')
                logger.info('Weights saved!')
                
            if self.config.render_training:
                self.env.render()
                
        episode_info = {
            'reward': episode_reward,
            'steps': steps,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(episode_loss) if episode_loss else 0,
        }
        
        if self.config.use_wandb:
            wandb.log({
                "episode/reward": episode_reward,
                "episode/length": steps,
                "episode/avg_loss": episode_info['avg_loss'],
                "episode/total_timesteps": self.total_timesteps
            }, step=self.total_timesteps)
            
        return episode_info

    def evaluate(self, num_episodes: int = 10) -> float:
        rewards = []
        self.model.eval()
        
        for i in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                if self.config.render_evaluation:
                    self.env.render()
                
                with torch.no_grad():
                    action = self.select_action(state, training=False)
                    state, reward, done, _ = self.env.step(action)
                    episode_reward += reward
            
            rewards.append(episode_reward)
            
            if self.config.use_wandb:
                wandb.log({
                    f"eval/episode_{i}_reward": episode_reward,
                }, step=self.total_timesteps)
        
        mean_reward = float(np.mean(rewards))
        if self.config.use_wandb:
            wandb.log({
                "eval/mean_reward": mean_reward,
                "eval/max_reward": max(rewards),
                "eval/min_reward": min(rewards),
                "eval/reward_std": float(np.std(rewards))
            }, step=self.total_timesteps)
        
        self.model.train()
        return mean_reward

    def save_checkpoint(self, path: Path) -> None:
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'total_timesteps': self.total_timesteps,
                'learns': self.learns,
            }
            torch.save(checkpoint, path)
            logger.info(f"Saved checkpoint to {path}")
            
            if self.config.use_wandb:
                artifact = wandb.Artifact(
                    name=f"model-checkpoint-{self.total_timesteps}", 
                    type="model"
                )
                artifact.add_file(str(path))
                wandb.log_artifact(artifact)
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(self, path: Path) -> None:
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.total_timesteps = checkpoint.get('total_timesteps', 0)
            self.learns = checkpoint.get('learns', 0)
            self.target_model.load_state_dict(self.model.state_dict())
            logger.info(f"Loaded checkpoint from {path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise