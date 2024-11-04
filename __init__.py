from gym.envs.registration import register

register(
    id='Shooter-v0',
    entry_point='envs:ShooterEnv',
)