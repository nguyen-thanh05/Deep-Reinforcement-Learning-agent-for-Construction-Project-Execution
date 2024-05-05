from gymnasium.envs.registration import register

register(
    id="ScaffoldGridWorld_env/ScaffoldGridWorld",
    entry_point="ScaffoldGridWorld_env.envs:ScaffoldGridWorldEnv",
)
