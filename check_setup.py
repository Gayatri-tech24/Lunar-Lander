import gymnasium as gym
import torch
import stable_baselines3

print("Gymnasium",gym.__version__)
print("Torch",torch.__version__)
print("Stable-baseline3",stable_baselines3.__version__)

env=gym.make("LunarLander-v3")
observation,info=env.reset(seed=42)

print("\nObservation:", observation)
print("Number of observations:", len(observation))
print("Available actions:", env.action_space.n)

env.close()
print("\nSetup successful!")