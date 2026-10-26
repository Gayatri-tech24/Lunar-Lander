
import gymnasium as gym
import time

env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset(seed=42)

total_reward = 0
terminated = False
truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    time.sleep(0.02)

print("Episode reward:", round(total_reward, 2))

env.close()
