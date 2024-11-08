import gym
import gym_donkeycar
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from concurrent.futures import ThreadPoolExecutor
from stable_baselines3.common.vec_env import VecNormalize, VecTransposeImage
from gym import spaces
from typing import Dict, Any, List
from dataclasses import dataclass, field

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

class EntropyDecayCallback(BaseCallback):
    """
    Callback for decaying the entropy coefficient over time.
    """
    def __init__(self, start_ent_coef: float, end_ent_coef: float, total_timesteps: int, verbose: int = 0):
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
    def __init__(self, total_timesteps: int, name: str, verbose=0):
        super().__init__(verbose)
        self.save_interval = total_timesteps // 10
        self.name = name
 
    def _on_step(self) -> bool:
        # Save the model every 'save_interval' steps
        if self.num_timesteps % self.save_interval == 0:
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
        "body_style": "cybertruck", # "donkey" | "bare" | "car01" | "f1" | "cybertruck"
        "body_rgb": (128, 128, 128),
        "car_name": name,
        "font_size": 20,
        "cam_resolution": cam,
        "cam_config" : {
            "img_w": cam[0],
            "img_h": cam[1],
            "img_d": cam[2],
            "fov" : 150, 
            "fish_eye_x" : 1.0,
            "fish_eye_y" : 1.0,
            "img_enc" : "PNG",
            "offset_x" : 0.0,
            "offset_y" : 0.0,
            "offset_z" : 0.0,
            "rot_x" : 0,
            "rot_y" : 0,
            "rot_z" : 0
        },
    }

    env = gym.make("donkey-waveshare-v0", conf=conf)
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)
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
    batch_size: int = 128
    n_steps: int = 2048
    gamma: float = 0.90
    learning_rate: float = 1e-4
    initial_entropy_coef: float = 0.05
    final_entropy_coef: float = 0.05
    vf_coef: float = 0.1
    max_grad_norm: float = 0.5
    n_epochs: int = 20
    net_arch: List[int] = field(default_factory=lambda: [512, 512, 256])
    clip_range: float = 0.3
    # normalize_advantage: bool = True
    target_kl: float = 0.15

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

    # Creating the environment
    env = create_env(config.port, config.model_name)
    
    # Initialize the PPO agent with tuned parameters
    model = PPO(
        "CnnPolicy",
        env,
        verbose=0,
        ent_coef=config.ppo.initial_entropy_coef,
        batch_size=config.ppo.batch_size,
        n_steps=config.ppo.n_steps,
        gamma=config.ppo.gamma,
        learning_rate=config.ppo.learning_rate,
        vf_coef=config.ppo.vf_coef,
        max_grad_norm=config.ppo.max_grad_norm,
        n_epochs=config.ppo.n_epochs,
        policy_kwargs=dict(net_arch=config.ppo.net_arch),
        clip_range=config.ppo.clip_range,
        # normalize_advantage=config.ppo.normalize_advantage,
        target_kl=config.ppo.target_kl
    )

    # Instantiating and training the DQN agent with callback for saving
    callbacks = [
        SaveOnIntervalCallback(config.total_timesteps, config.model_name),
        EntropyDecayCallback(config.ppo.initial_entropy_coef, config.ppo.final_entropy_coef, config.total_timesteps)
    ]

    # Resetting the environment
    obs = env.reset()
    
    # Start training the models
    try:
        model.learn(config.total_timesteps, callback=callbacks)
    finally:
        env.close()

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
        TrainingConfig(
            model_name="default",
            total_timesteps=100_000,
            port=9092,
        ),
        TrainingConfig(
            model_name="shorty",
            total_timesteps=50_000,
            port=9093,
        ),
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        train_futures = [executor.submit(train_model, config) for config in train_configs]
        for future in train_futures:
            future.result()
        
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

    with ThreadPoolExecutor(max_workers=3) as executor:
        train_futures = [executor.submit(evaluate_model, config) for config in train_configs]
        for future in train_futures:
            future.result()
        
    print("Done!")