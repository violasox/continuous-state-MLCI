from garage.envs import GymEnv

class GymEnvWithMeta(GymEnv):
    def reset(self):
        first_obs = self._env.reset()
        episode_metadata = self._env.get_episode_info()
        self._step_cnt = 0
        self._env.info = None
        return (first_obs, episode_metadata)
