# LunarLander-v3 with PPO (Stable-Baselines3)

Training a reinforcement learning agent to land a spacecraft in Gymnasium's `LunarLander-v3`, comparing a random baseline, a standard PPO agent, and PPO trained with custom reward shaping.

![Learning curve](lunar.mp4)

<!-- Replace with your GIF/video once recorded: ![Landing demo](demo.gif) -->

## Results

All scores are the mean ± standard deviation of total episode reward over 100 evaluation episodes, using the **original** environment reward. A score of 200 or more is considered "solved".

| Agent | Mean reward | Std |
|---|---|---|
| Random actions | XX.X | XX.X |
| PPO (default reward) | XXX.X | XX.X |
| PPO (custom reward) | XXX.X | XX.X |

## Approach

- **Algorithm:** PPO with an MLP policy (Stable-Baselines3)
- **Training:** 16 parallel environments, 1,000,000 timesteps
- **Key hyperparameters:** learning rate 3e-4, `n_steps` 1024, batch size 64, 4 epochs, gamma 0.999, GAE lambda 0.98, entropy coefficient 0.01
- **Model selection:** `EvalCallback` evaluates periodically and keeps the best checkpoint
- **Seed:** 42 for training (results vary by seed, see notes below)

### Custom reward experiment

LunarLander's default reward is already shaped (distance to pad, speed, angle, leg contact, fuel use, +100 landing / -100 crash). I wrapped the environment to add extra terms and trained on the shaped reward, while **evaluating on the original reward** so scores stay comparable.

Terms I tried:

- **Spin penalty:** `-k * |angular velocity|` to discourage tumbling
- **Soft descent bonus:** bonus for low, slow descent near the ground
- **Step penalty:** small constant cost per step to discourage hovering

**What happened:** <!-- Fill in honestly, e.g. "The descent bonus led the agent to hover near the ground to collect bonus instead of landing (a reward-hacking behaviour). The spin penalty alone gave a small / no improvement over the baseline." -->

**Takeaway:** <!-- e.g. "Reward shaping needs to be tested one term at a time, and the evaluation metric must stay fixed." -->

## Project structure

```
watch_lander.py          # random agent, renders a window (baseline behaviour)
train_ppo.py             # trains PPO with the default reward
train_custom_reward.py   # trains PPO with the custom reward wrapper
watch_trained.py         # renders the trained agent
evaluate_and_record.py   # 100-episode evaluation, learning curve, video recording
```

## Setup

```bash
pip install "gymnasium[box2d]" stable-baselines3 matplotlib moviepy
```

## Usage

```bash
python train_ppo.py              # train baseline, saves best_model/best_model.zip
python train_custom_reward.py    # train with custom reward, saves best_model_custom/
python watch_trained.py          # watch the trained agent
python evaluate_and_record.py    # print scores, save learning_curve.png and videos/
```

To evaluate the custom-reward model, change `MODEL_PATH` in `evaluate_and_record.py` to `best_model_custom/best_model.zip` and the log path to `logs_custom/evaluations.npz`.

## Notes and limitations

- RL results are sensitive to the random seed. Results above come from seed(s): <!-- e.g. 42, 43, 44 -->.
- Evaluation uses deterministic actions on 100 episodes; a different set of seeds may give slightly different numbers.
- The environment is simple, so this project is mainly about the training and evaluation workflow rather than state-of-the-art performance.

## Possible next steps

- Compare against DQN and A2C
- Average results over multiple seeds
- Try the continuous action version (`continuous=True`) with SAC or TD3
