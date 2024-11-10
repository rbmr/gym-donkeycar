import gym
import gym_donkeycar
import numpy as np
from stable_baselines.ppo2 import PPO2 as PPO
from stable_baselines.common.callbacks import BaseCallback
import os 
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from concurrent.futures import ThreadPoolExecutor
from stable_baselines.common.vec_env import VecNormalize
from gym import spaces
from typing import Dict, Any, List
from dataclasses import dataclass, field
import cv2
from typing import Optional

# Creating directories for storing logs and models
pitcrew_dir = os.path.dirname(__file__)
log_dir = os.path.join(pitcrew_dir, "pitcrew_logs") # Directory for storing training logs
models_dir = os.path.join(pitcrew_dir, "pitcrew_models") # Directory for storing models

model_name_steps = lambda name, timesteps: f'model{"_" if name else ""}{name}_{timesteps}'
model_name_steps_zip = lambda name, timesteps: f"{model_name_steps(name, timesteps)}.zip"
dir_model_name_steps = lambda name, timesteps: os.path.join(models_dir, model_name_steps(name, timesteps))
dir_model_name_steps_zip = lambda name, timesteps: os.path.join(models_dir, model_name_steps_zip(name, timesteps))

# Ensuring the directories exist or creating them
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

yellow_ref_hue = 40
crop_top_percent = 0.40

def compute_yellow_score(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to yellow detection score (0-1 scale).
    Args: img: RGB image array of shape (H, W, 3)
    Returns: Array of shape (H, W) with values 0-1 indicating yellow intensity
    """
    # Convert RGB to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    h = h.astype(np.int16) 
    h = np.abs(h - yellow_ref_hue)
    hue_diff = np.minimum(h, 180 - h)
    hue_score = 1 - (hue_diff / 90.0)

    # Weighted average
    yellow_score = hue_score * (7/16) + (s / 255.0) * (5/16) + (v / 255.0) * (4/16) 
    yellow_score *= yellow_score
    yellow_score /= np.max(yellow_score)

    yellow_score_uint8 = (yellow_score * 255).astype(np.uint8)
    threshold, _ = cv2.threshold(yellow_score_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return (yellow_score_uint8 >= threshold).astype(np.uint8)

def preprocess(obs: np.ndarray) -> np.ndarray:
    """
    Apply all preprocessing steps to observation.
    """
    if obs is None:
        return None
        
    # Crop top portion
    crop_height = int(obs.shape[0] * crop_top_percent)
    cropped = obs[crop_height:, :, :]
    
    # Convert to yellow score
    yellow_score = compute_yellow_score(cropped)
    
    return yellow_score[..., np.newaxis]
    
def preprocess_obs_space(original_space: spaces.Box) -> spaces.Box:
    """
    Calculate the new observation space after preprocessing.
    New observation space
    """
    orig_shape = original_space.shape
    crop_height = int(orig_shape[0] * crop_top_percent)
    new_height = orig_shape[0] - crop_height
    
    return spaces.Box(
        low=0,
        high=1,
        shape=(new_height, orig_shape[1], 1),
        dtype=np.uint8
    )

class PreprocessingEnv(gym.Wrapper):
    """Environment wrapper that applies preprocessing to observations."""
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = preprocess_obs_space(env.observation_space)
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Preprocess the observation."""
        return preprocess(obs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)

class TrainingVisualizer:
    """Helper class to visualize training observations without impacting performance."""
    
    def __init__(self, window_name: str = "Training View", enabled: bool = True):
        self.window_name = window_name
        self.enabled = enabled
        if self.enabled:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 640, 480)
    
    def show(self, frame: np.ndarray) -> None:
        """Display the current frame if visualization is enabled."""
        if not self.enabled or frame is None:
            return
            
        # For single channel images
        if len(frame.shape) == 3 and frame.shape[-1] == 1:
            frame = frame[:, :, 0]
        
        # Convert to display format
        frame_display = (frame * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_display, cv2.COLOR_GRAY2BGR)
        
        # Show frame
        cv2.imshow(self.window_name, frame_bgr)
        cv2.waitKey(1)

    
    def close(self) -> None:
        """Clean up the visualization window."""
        if self.enabled:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)  # Give time for window to actually close

    def __del__(self):
        """Ensure window is closed on deletion."""
        self.close()

class VisualizationCallback(BaseCallback):
    def __init__(self, visualizer: TrainingVisualizer, verbose=0):
        super().__init__(verbose)
        self.visualizer = visualizer
        
    def _on_step(self):
        # Get the model's actual observation (this will be cropped)
        if isinstance(self.training_env, DummyVecEnv):
            env = self.training_env.envs[0]
        else:
            env = self.training_env
    
        if isinstance(env, PreprocessingEnv):
            # Get the raw image and apply cropping
            raw_obs = env.viewer.handler.image_array
            obs = env.observation(raw_obs)
        else:
            obs = env.viewer.handler.image_array
            
        self.visualizer.show(obs)
        return True

class EntropyDecayCallback(BaseCallback):
    """
    Callback for decaying the entropy coefficient over time.
    """
    def __init__(self, start_ent_coef: float, end_ent_coef: float, total_timesteps: int, verbose: int = 2):
        super().__init__(verbose)
        self.start_ent_coef = start_ent_coef
        self.end_ent_coef = end_ent_coef
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Linear decay of entropy coefficient
        progress = self.num_timesteps / self.total_timesteps
        current_ent_coef = self.start_ent_coef - (self.start_ent_coef - self.end_ent_coef) * progress
        self.model.ent_coef = current_ent_coef
        return True

class SaveOnIntervalCallback(BaseCallback):
    """
    Callback for saving models at regular intervals during training
    """
    def __init__(self, save_interval: int, name: str, verbose=2):
        super().__init__(verbose)
        self.save_interval = save_interval
        self.name = name

    def _on_step(self) -> bool:
        # Save the model every 'save_interval' steps
        if (self.num_timesteps-1) % self.save_interval == 0:
            self.model.save(dir_model_name_steps(self.name, self.num_timesteps))
            if self.verbose > 0:
                print(f'Saving model to {model_name_steps_zip(self.name, self.num_timesteps)}')
        return True

def create_env(port: int=9091, name: str = "pitcrew") -> gym.Env:
    """Create and configure the environment."""

    cam = (224, 224, 3)

    conf = {
        "exe_path": "/home/robert/projects/DonkeySimLinux/donkey_sim.x86_64",
        "host": "127.0.0.1",
        "port": port,
        "body_style": "donkey", # "donkey" | "bare" | "car01" | "f1" | "cybertruck"
        "body_rgb": (128, 128, 128),
        "car_name": name,
        "font_size": 20,
        "cam_resolution": cam,
        "cam_config" : {
            "img_w": cam[0],
            "img_h": cam[1],
            "img_d": cam[2],
            "fov" : 100, 
            "fish_eye_x" : 0.5,
            "fish_eye_y" : 0.5,
            "img_enc" : "PNG", # "PNG"
            "offset_x" : 0.0,
            "offset_y" : 0.0,
            "offset_z" : 0.0,
            "rot_x" : 0,
            "rot_y" : 0,
            "rot_z" : 0
        },
    }


    env = gym.make("donkey-waveshare-v0", conf=conf)
    env = PreprocessingEnv(env)
    return env

@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""
    model_name: str = None
    model_steps: int = 100_000
    evaluate_timesteps: int = 10_000
    port: int = 9091

@dataclass
class PPOConfig:
    """Hyperparameters specific to PPO algorithm."""
    # batch_size: int = 128
    nminibatches: int = 8
    n_steps: int = 2048
    gamma: float = 0.90
    learning_rate: float = 1e-4
    initial_entropy_coef: float = 0.01
    final_entropy_coef: float = 0.005
    vf_coef: float = 0.1
    max_grad_norm: float = 0.5
    # n_epochs: int = 20
    noptepochs: int = 10
    net_arch: List[int] = field(default_factory=lambda: [256, 128, 64])
    # clip_range: float = 0.3
    cliprange: float = 0.2
    # normalize_advantage: bool = True
    # target_kl: float = 0.15

@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    model_name: str = None
    total_timesteps: int = 100_000
    port: int = 9091
    ppo: PPOConfig = field(default_factory=PPOConfig)

def evaluate_model(config: EvalConfig = None):

    if config is None:
        config = EvalConfig()

    # Creating the environment
    env = create_env(config.port, config.model_name)

    # Load the PPO agent
    model = PPO.load(dir_model_name_steps_zip(config.model_name, config.model_steps))

    # Resetting the environment
    obs = env.reset()

    # Run the model in inference mode
    total_reward = 0
    for _ in range(config.evaluate_timesteps):  # Adjust the range as needed
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            obs = env.reset()
    
    env.close()

    # compute average reward per thousand timesteps
    reward_per_ktimesteps = total_reward / config.evaluate_timesteps * 1000
    
    print(f"Evaluation of {model_name_steps(config.model_name, config.model_steps)} finished.")
    print(f"Number of evaluation timesteps: {config.evaluate_timesteps}")
    print(f"Average reward per 1000 timesteps was: {reward_per_ktimesteps}")
    
    return reward_per_ktimesteps

def train_model(config: TrainingConfig = None):

    if config is None:
        config = TrainingConfig()
    
    try:
        # Create camera viewer
        vis = TrainingVisualizer(enabled=True)

        # Creating the environment
        env = create_env(config.port, config.model_name)

        # Initialize the PPO agent with tuned parameters (for stable baselines 2)
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            ent_coef=config.ppo.initial_entropy_coef,
            # batch_size=config.ppo.batch_size,
            nminibatches=config.ppo.nminibatches,
            n_steps=config.ppo.n_steps,
            gamma=config.ppo.gamma,
            learning_rate=config.ppo.learning_rate,
            vf_coef=config.ppo.vf_coef,
            max_grad_norm=config.ppo.max_grad_norm,
            # n_epochs=config.ppo.n_epochs,
            noptepochs=config.ppo.noptepochs,
            policy_kwargs=dict(net_arch=[{'pi': config.ppo.net_arch, 'vf': config.ppo.net_arch}]),
            # clip_range=config.ppo.clip_range,
            cliprange=config.ppo.cliprange,
            # normalize_advantage=config.ppo.normalize_advantage,
            # target_kl=config.ppo.target_kl
        )

        # Instantiating and training the agent with callback for saving
        callbacks = [
            VisualizationCallback(vis),
            SaveOnIntervalCallback(save_interval=config.ppo.n_steps, name=config.model_name),
            EntropyDecayCallback(config.ppo.initial_entropy_coef, config.ppo.final_entropy_coef, config.total_timesteps)
        ]

        # Resetting the environment
        obs = env.reset()

        # Start training the models
        model.learn(config.total_timesteps, callback=callbacks)

    finally:
        if 'env' in locals():
            env.close()
        if 'vis' in locals():
            vis.close()

    return model

if __name__ == "__main__":

    train_configs = [
        TrainingConfig(
            model_name="edecay",
            total_timesteps=100_000,
            port=9091,
            ppo=PPOConfig(
                initial_entropy_coef=0.05,
                final_entropy_coef=0.01,
            )
        ),
        # TrainingConfig(
        #     model_name="default",
        #     total_timesteps=100_000,
        #     port=9092,
        # ),
        # TrainingConfig(
        #     model_name="shorty",
        #     total_timesteps=50_000,
        #     port=9093,
        # ),
    ]

    for config in train_configs:
        train_model(config)
        
    print("All training completed. Starting evaluation...")

    eval_configs = [
        EvalConfig(
            model_name = config.model_name,
            model_steps = config.total_timesteps,
            port = config.port,
            evaluate_timesteps = 10_000
        )
        for config in train_configs
    ]

    for config in eval_configs:
        evaluate_model(config)
        
    print("Done!")