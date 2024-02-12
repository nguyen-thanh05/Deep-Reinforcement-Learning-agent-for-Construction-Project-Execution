from gymnasium.envs.registration import register

register(
    id="SharedGridWorld_env/SharedGridWorld",
    entry_point="SharedGridWorld_env.envs:SharedGridWorldEnv",
)
