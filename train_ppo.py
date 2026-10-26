
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

train_env = gym.make("LunarLander-v3")
eval_env = gym.make("LunarLander-v3")

# Evaluate the model every 10,000 steps
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
    verbose=1,
)

# Start a fresh model with adjusted PPO settings
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
    seed=43,
)

print("Starting fresh PPO training...")

model.learn(
    total_timesteps=300_000,
    callback=eval_callback,
)

model.save("lander_v2")

train_env.close()
eval_env.close()

print("Training complete! Saved as lander_v2.zip")
print("The best evaluation model is in best_model/")
