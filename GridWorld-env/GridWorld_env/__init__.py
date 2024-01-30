from gymnasium.envs.registration import register

register(
    id="GridWorld_env/GridWorld",
    entry_point="GridWorld_env.envs:GridWorldEnv",
)
