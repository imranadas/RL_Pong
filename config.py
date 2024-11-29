import yaml
import logging
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class TrainingConfig(BaseModel):
    # Model parameters
    hidden_size: int = Field(default=512, gt=0)
    n_input_channels: int = Field(default=4, gt=0)
    
    # Training parameters
    buffer_size: int = Field(default=75000, gt=0)
    batch_size: int = Field(default=32, gt=0)
    target_update: int = Field(default=1000, gt=0)
    train_episodes: int = Field(default=10000, gt=0)
    eval_episodes: int = Field(default=10, gt=0)
    learning_rate: float = Field(default=0.0001, gt=0)
    gamma: float = Field(default=0.95, ge=0, le=1)
    frame_stack: int = Field(default=4, gt=0)
    starting_mem_len: int = Field(default=50000, gt=0)
    
    # Exploration parameters
    epsilon_start: float = Field(default=1.0, ge=0, le=1)
    epsilon_final: float = Field(default=0.05, ge=0, le=1)
    epsilon_decay: float = Field(default=0.9/100000, gt=0)
    
    # DQN variants
    double_dqn: bool = True
    dueling_dqn: bool = True
    
    # Training features
    use_wandb: bool = True
    render_training: bool = False
    render_evaluation: bool = True
    
    # Paths
    checkpoint_dir: Path = Field(default=Path("checkpoints"))
    load_checkpoint: Optional[Path] = None

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "TrainingConfig":
        try:
            with open(yaml_path, "r") as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
        except (yaml.YAMLError, FileNotFoundError) as e:
            logger.error(f"Failed to load config from {yaml_path}: {e}")
            raise

    def save_yaml(self, yaml_path: Path) -> None:
        try:
            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            with open(yaml_path, "w") as f:
                yaml.dump(self.model_dump(), f)
        except Exception as e:
            logger.error(f"Failed to save config to {yaml_path}: {e}")
            raise