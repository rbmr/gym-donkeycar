import gym
import gym_donkeycar
import numpy as np
from stable_baselines3 import PPO

env = gym.make("donkey-waveshare-v0")

model = PPO("CnnPolicy", env, n_steps=256, verbose=1)

obs = env.reset()

model.learn(50_000)


env.close()

# to be replaced by the RL zoo