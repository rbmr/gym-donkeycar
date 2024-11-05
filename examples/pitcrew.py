import gym
import gym_donkeycar
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
from stable_baselines3.common.monitor import Monitor

# Define a callback class for saving models at regular intervals during training
class SaveOnIntervalCallback(BaseCallback):
    def __init__(self, save_interval: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_interval = save_interval
        self.save_path = save_path
 
    def _on_step(self) -> bool:
        # Save the model every 'save_interval' steps
        if self.num_timesteps % self.save_interval == 0:
            save_file = os.path.join(self.save_path, f'model_{self.num_timesteps}')
            self.model.save(save_file)
            if self.verbose > 0:
                print(f'Saving model to {save_file}.zip')
        return True

# Creating directories for storing logs and models
log_dir = "pitcrew_logs"  # Directory for storing training logs
models_dir = "pitcrew_models"  # Directory for storing models

# Ensuring the directories exist or creating them
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Creating and wrapping the environment with a monitor for logging
env = gym.make("donkey-waveshare-v0")
# env = Monitor(env, log_dir) 

# Instantiating and training the DQN agent with callback for saving
total_timesteps = 100_000
save_interval = total_timesteps // 10
save_callback = SaveOnIntervalCallback(save_interval, models_dir)

# Initialize the PPO agent
model = PPO("CnnPolicy", env, n_steps=256, verbose=1)

# Resetting the environment
obs = env.reset()

# Start training the models
model.learn(total_timesteps, callback=save_callback)

# Save the final model
model.save(os.path.join(models_dir, "model_final"))

# Close the environment properly.
env.close()