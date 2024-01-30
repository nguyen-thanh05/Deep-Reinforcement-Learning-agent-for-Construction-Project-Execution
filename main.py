
import gymnasium as gym
import GridWorld_env
from gymnasium.wrappers import FlattenObservation


env = gym.make('GridWorld_env/GridWorld', render_mode="human")
#env = gym.make("LunarLander-v2", render_mode="human")
env.action_space.seed(42)

wrapped_env = FlattenObservation(env)
print(wrapped_env.reset())     # E.g.  [3 0 3 3], {}
observation, info = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation, info = env.reset()

env.close()

