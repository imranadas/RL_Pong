import torch
import logging
import numpy as np
import torch.nn as nn
from config import TrainingConfig
from typing import Tuple, NamedTuple

logger = logging.getLogger(__name__)

class Transition(NamedTuple):
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool

class DuelingDQN(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.n_actions = 3
        
        self.conv = nn.Sequential(
            nn.Conv2d(config.frame_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        conv_out_size = self._get_conv_out_size(config.frame_stack)
        
        if config.dueling_dqn:
            self.advantage = nn.Sequential(
                nn.Linear(conv_out_size, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(config.hidden_size, self.n_actions)
            )
            
            self.value = nn.Sequential(
                nn.Linear(conv_out_size, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(config.hidden_size, 1)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(conv_out_size, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(config.hidden_size, self.n_actions)
            )
        
        self.dueling = config.dueling_dqn
        self._init_weights()
        
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def _get_conv_out_size(self, input_channels: int) -> int:
        dummy_input = torch.zeros(1, input_channels, 84, 84)
        conv_out = self.conv(dummy_input)
        return int(np.prod(conv_out.shape[1:]))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle single state input (during evaluation)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Now x should be [batch_size, frames, h, w]
        # We don't need to add a channel dimension since we're using frame_stack as channels
        conv_out = self.conv(x)
        flattened = conv_out.view(conv_out.size(0), -1)
        
        if self.dueling:
            advantage = self.advantage(flattened)
            value = self.value(flattened)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            return self.fc(flattened)

class PrioritizedReplayBuffer:
    def __init__(self, config: TrainingConfig, device: torch.device):
        self.capacity = config.buffer_size
        self.device = device
        self.transitions: list[Transition] = []
        self.position = 0
        
    def push(self, state: torch.Tensor, action: int, reward: float, 
             next_state: torch.Tensor, done: bool) -> None:
        transition = Transition(state, action, reward, next_state, done)
        
        if len(self.transitions) < self.capacity:
            self.transitions.append(transition)
        else:
            self.transitions[self.position] = transition
            
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.transitions), batch_size)
        batch = [self.transitions[idx] for idx in indices]
        
        return (
            torch.stack([t.state for t in batch]).to(self.device),
            torch.tensor([t.action for t in batch], dtype=torch.long).to(self.device),
            torch.tensor([t.reward for t in batch], dtype=torch.float).to(self.device),
            torch.stack([t.next_state for t in batch]).to(self.device),
            torch.tensor([t.done for t in batch], dtype=torch.float).to(self.device),
        )
    
    def __len__(self) -> int:
        return len(self.transitions)