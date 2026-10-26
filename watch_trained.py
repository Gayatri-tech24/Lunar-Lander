
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3", render_mode="human")
model = PPO.load("best_model/best_model.zip")

for episode in range(5):
    obs, info = env.reset(seed=episode + 100)
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print(f"Episode {episode + 1}: reward = {total_reward:.2f}")

env.close()
