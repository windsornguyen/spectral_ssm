import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import numpy as np


# Function to create the environment
def make_env():
    env = gym.make("Ant-v4")
    env = Monitor(env)
    env = gym.wrappers.rescale_action.RescaleAction(env, min_action=-1, max_action=1)
    return env


# Create the environment for training
env = DummyVecEnv([make_env])

# Initialize the PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=None,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./ppo_ant_tensorboard/",
)

# Train the agent
total_timesteps = 10000
model.learn(total_timesteps=total_timesteps)

# Save the trained model
model.save("ppo_ant")

# Load the trained model
loaded_model = PPO.load("ppo_ant")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Create a new environment for rendering and video recording
render_env = gym.make(
    "Ant-v4", render_mode="rgb_array", camera_name="free", max_episode_steps=1000
)
render_env = Monitor(render_env)
render_env = gym.wrappers.rescale_action.RescaleAction(
    render_env, min_action=-1, max_action=1
)
render_env = gym.wrappers.RecordVideo(render_env, video_folder="./videos/")

# Run and record episodes
num_episodes = 5
for episode in range(num_episodes):
    print(f"Episode {episode + 1}")
    obs, _ = render_env.reset()
    episode_reward = 0.0
    for _ in range(1000):  # Max 1000 steps per episode
        action, _states = loaded_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = render_env.step(action)
        episode_reward += float(reward)
        if terminated or truncated:
            break
    print(f"Episode reward: {episode_reward:.2f}")

# Close the environment
render_env.close()

print(
    "Training and evaluation complete. Check the './videos/' directory for recorded episodes."
)
