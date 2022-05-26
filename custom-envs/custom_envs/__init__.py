from gym.envs.registration import register
from custom_envs.gym_env_meta import GymEnvWithMeta
from custom_envs.categorical_policy import CategoricalPolicy
from os import path

ROBOTS_DIR = path.join(path.dirname(__file__), 'robots')
EXP5_FILE = path.join(path.dirname(__file__), 'startsEnds5.pkl')
EXP8_FILE = path.join(path.dirname(__file__), 'startsEnds8.pkl')

register(
    id='InvertedPendulumDiscreteActionEnv-v0',
    entry_point='custom_envs.envs:InvertedPendulumDiscreteActionEnv',
    max_episode_steps=2000,
    reward_threshold=100,
)
