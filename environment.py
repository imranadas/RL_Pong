import cv2
import torch
import logging
import numpy as np
import gymnasium as gym
from typing import Tuple
from collections import deque

logger = logging.getLogger(__name__)

class FrameStack:
    def __init__(self, frames):
        self.buffer = deque(maxlen=frames)
        
    def push(self, frame):
        # frame should be shape [1, 1, 84, 84]
        self.buffer.append(frame)
        
    def get(self):
        # Stack frames along channel dimension and remove extra dimensions
        stacked = torch.cat(list(self.buffer), dim=1)  # [1, frames, 84, 84]
        return stacked.squeeze(0)  # Return [frames, 84, 84]
        
    def clear(self):
        self.buffer.clear()
        
class PongEnvironment:
    def __init__(self, config):
        render_mode = "rgb_array"
        
        self.env = gym.make("ALE/Pong-v5", 
                          render_mode=render_mode,
                          frameskip=1,
                          repeat_action_probability=0.0,
                          full_action_space=False,
                          obs_type='grayscale',
                          max_episode_steps=10000)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frame_stacker = FrameStack(config.frame_stack)
        logger.info(f"Initialized Fast PongEnvironment with device: {self.device}")
        self.possible_actions = [0, 2, 3]
        
        self.frame_skip = 2
        
    def preprocess(self, frame):
        # Crop and resize
        frame = frame[30:-12, 5:-4]
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Convert to tensor and normalize
        frame_tensor = torch.from_numpy(frame).float().to(self.device) / 255.0
        
        # Add dimensions to make it [1, 1, 84, 84]
        return frame_tensor.unsqueeze(0).unsqueeze(0)

    def reset(self) -> torch.Tensor:
        obs, _ = self.env.reset()
        frame = self.preprocess(obs)
        
        self.frame_stacker.clear()
        for _ in range(self.frame_stacker.buffer.maxlen):
            self.frame_stacker.push(frame)
            
        return self.frame_stacker.get()

    def step(self, action_idx: int) -> Tuple[torch.Tensor, float, bool, dict]:
        total_reward = 0
        done = False
        info = {}
        
        action = self.possible_actions[action_idx]
        for _ in range(self.frame_skip):
            if not done:
                obs, reward, terminated, truncated, step_info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
                info = step_info
        
        frame = self.preprocess(obs)
        self.frame_stacker.push(frame)
        
        return self.frame_stacker.get(), total_reward, done, info

    def render(self) -> np.ndarray:
        return self.env.render() if self.env.render_mode == "rgb_array" else None

    def close(self) -> None:
        try:
            self.env.close()
            cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"Error closing environment: {e}")

    @property
    def action_space(self) -> int:
        return len(self.possible_actions)