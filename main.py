import torch
import wandb
import typer
import logging
import numpy as np
import pandas as pd
from tqdm import trange
from pathlib import Path
from datetime import datetime
from trainer import DQNTrainer
from config import TrainingConfig
from rich.logging import RichHandler
from environment import PongEnvironment
from model import DuelingDQN, PrioritizedReplayBuffer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger(__name__)

def initialize_wandb(config: TrainingConfig, mode: str):
    """Initialize WandB run with proper configuration"""
    run_name = f"pong-dqn-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="pong-dqn",
        name=run_name,
        mode=mode,
        config={
            "model_type": "DuelingDQN" if config.dueling_dqn else "DQN",
            "double_dqn": config.double_dqn,
            "train_episodes": config.train_episodes,
            "eval_episodes": config.eval_episodes,
            "buffer_size": config.buffer_size,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "frame_stack": config.frame_stack,
            "epsilon_start": config.epsilon_start,
            "epsilon_final": config.epsilon_final,
            "epsilon_decay": config.epsilon_decay
        }
    )
    logger.info(f"Initialized WandB run: {run_name}")

def format_metrics(metrics: dict) -> str:
    """Format metrics for tqdm display"""
    return f"ε: {metrics['epsilon']:.3f}, R: {metrics['reward']:.1f}, Loss: {metrics['avg_loss']:.4f}"

def main(
    config_path: Path = typer.Option("config.yml", "--config", "-c", help="Path to config file"),
    checkpoint_dir: Path = typer.Option("checkpoints", "--checkpoint-dir", "-d", help="Directory for checkpoints"),
    test_mode: bool = typer.Option(False, "--test", help="Run in test mode with loaded weights"),
    wandb_mode: str = typer.Option("online", "--wandb-mode", help="WandB mode (online/offline/disabled)")
) -> None:
    """Train a DQN agent to play Pong with WandB logging"""
    try:
        # Load config
        config = TrainingConfig.from_yaml(config_path)
        config.checkpoint_dir = checkpoint_dir
        
        # Initialize WandB if enabled
        if config.use_wandb and wandb_mode != "disabled":
            initialize_wandb(config, wandb_mode)
        
        # Create checkpoint directory
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Initialize components and trainer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = PongEnvironment(config)
        model = DuelingDQN(config).to(device)
        target_model = DuelingDQN(config).to(device)
        buffer = PrioritizedReplayBuffer(config, device)
        
        trainer = DQNTrainer(
            env=env,
            model=model,
            target_model=target_model,
            buffer=buffer,
            device=device,
            config=config
        )
        
        if config.load_checkpoint or test_mode:
            checkpoint_path = config.load_checkpoint or (checkpoint_dir / 'best_model.pth')
            trainer.load_checkpoint(checkpoint_path)
            if test_mode:
                trainer.epsilon = 0.0
                logger.info("Running in test mode with epsilon=0")

        # Training metrics tracking
        all_rewards = []
        best_eval_reward = float('-inf')
        running_reward = 0
        
        try:
            # Main training loop with tqdm
            pbar = trange(config.train_episodes, desc="Training", unit="episode")
            for episode in pbar:
                # Train one episode
                episode_info = trainer.train_episode()
                all_rewards.append(episode_info['reward'])
                
                # Update progress bar with current metrics
                pbar.set_postfix_str(format_metrics(episode_info))
                
                # Periodic evaluation
                if episode % 100 == 0:
                    eval_rewards = []
                    
                    # Evaluation loop with its own progress bar
                    logger.info("\nStarting evaluation...")
                    eval_pbar = trange(config.eval_episodes, desc="Evaluating", unit="episode", leave=False)
                    for eval_ep in eval_pbar:
                        eval_reward = trainer.evaluate(1)
                        eval_rewards.append(eval_reward)
                        eval_pbar.set_postfix_str(f"Reward: {eval_reward:.1f}")
                    
                    mean_eval_reward = np.mean(eval_rewards)
                    logger.info(f"Evaluation complete. Mean reward: {mean_eval_reward:.2f}")
                    
                    # Log metrics
                    if config.use_wandb:
                        wandb.log({
                            "eval/mean_reward": mean_eval_reward,
                            "eval/max_reward": max(eval_rewards),
                            "eval/min_reward": min(eval_rewards),
                            "eval/std_reward": np.std(eval_rewards),
                            "checkpoint": episode
                        }, step=trainer.total_timesteps)
                    
                    # Save if best model
                    if mean_eval_reward > best_eval_reward:
                        best_eval_reward = mean_eval_reward
                        trainer.save_checkpoint(checkpoint_dir / 'best_model.pth')
                        logger.info(f"New best model! Reward: {mean_eval_reward:.2f}")
                    
                    # Save periodic checkpoint
                    checkpoint_path = checkpoint_dir / f'model_episode_{episode}.pth'
                    trainer.save_checkpoint(checkpoint_path)
                    
                    # Log model artifact to WandB
                    if config.use_wandb:
                        model_artifact = wandb.Artifact(
                            f"model-checkpoint-{episode}", 
                            type="model",
                            description=f"Model checkpoint at episode {episode}"
                        )
                        model_artifact.add_file(str(checkpoint_path))
                        wandb.log_artifact(model_artifact)
                    
                    # Save rewards history
                    pd.DataFrame({'reward': all_rewards}).to_csv(
                        checkpoint_dir / 'rewards.csv', index=False
                    )

        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user")
            trainer.save_checkpoint(checkpoint_dir / 'interrupted_model.pth')
            
        finally:
            env.close()
            if config.use_wandb:
                wandb.finish()
            logger.info("Training completed")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        if config.use_wandb:
            wandb.finish(exit_code=1)
        raise

if __name__ == "__main__":
    typer.run(main)