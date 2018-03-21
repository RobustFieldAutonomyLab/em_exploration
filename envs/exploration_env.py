import math

import gym
from gym import error, spaces
from gym.utils import seeding

try:
    import ss2d
    from utils import load_config
except ImportError as e:
    raise error.DependencyNotInstalled('{}. Build em_exploration and export PYTHONPATH=build_dir'.format(e))


class ExplorationEnv(gym.Env):
    metadata = {'render.modes' : ['human', 'rgb_array']}

    def __init__(self, config):
        self._config = load_config(config)

        self.action_space = spaces.Box(low=-math.pi, high=math.pi)

        self._vm_cov_length = spaces.Box(low=-a, high=a, shape=())
        self._vm_cov_angle = spaces.Box(low=-math.pi, high=math.pi, shape=())
        self._vm_prob = spaces.Box(low=0.0, high=1.0, shape=())
        self.observation_space = spaces.Tuple()
        pass
    
    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass

if __name__ == '__main__':
        