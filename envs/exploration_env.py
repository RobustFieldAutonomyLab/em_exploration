import math
import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import error, spaces
from gym.utils import seeding

try:
    from pyplanner2d import EMExplorer
    from utils import load_config, plot_virtual_map, plot_virtual_map_cov
except ImportError as e:
    raise error.DependencyNotInstalled('{}. Build em_exploration and export PYTHONPATH=build_dir'.format(e))


class ExplorationEnv(gym.Env):
    metadata = {'render.modes' : ['human', 'state'],
                'render.pause' : 0.001}

    def __init__(self,
        config,  # exploration config file
        max_steps,  # termination
        ):
        self._config = load_config(config)
        self._max_steps = max_steps

        self.seed()
        self.reset()
        self._viewer = None

        num_actions = self._sim._planner_params.num_actions
        self._step_length = self._sim._planner_params.max_edge_length
        self._rotation_set = np.arange(0, np.pi * 2, np.pi * 2 / num_actions) - np.pi
        self._action_set = [np.array([np.cos(t) * self._step_length,
                                      np.sin(t) * self._step_length,
                                      t])
                            for t in self._rotation_set]
        self.action_space = spaces.Discrete(n=num_actions)
        assert(len(self._action_set) == num_actions)

        rows, cols = self._sim._virtual_map.to_array().shape
        self._max_sigma = self._sim._virtual_map.get_parameter().sigma0
        self._vm_cov_sigma = spaces.Box(low=0, high=self._max_sigma, shape=(rows, cols))
        self._vm_cov_angle = spaces.Box(low=-math.pi, high=math.pi, shape=(rows, cols))
        self._vm_prob = spaces.Box(low=0.0, high=1.0, shape=(rows, cols))
        self.observation_space = spaces.Tuple([self._vm_prob,
                                               self._vm_cov_sigma,
                                               self._vm_cov_angle])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        cov_array = self._sim._virtual_map.to_cov_array()
        self._obs = self._sim._virtual_map.to_array(), cov_array[0], cov_array[1]
        return self._obs

    def _get_utility(self, action=None):
        if action is None:
            distance = 0.0
        else:
            angle_weight = self._config.getfloat('Planner', 'angle_weight')
            distance = math.sqrt(self._step_length ** 2 +\
                                 (angle_weight * self._rotation_set[action]) ** 2)
        return self._sim.calculate_utility(distance)

    def step(self, action):
        u1 = self._get_utility()
        if self._sim.simulate(self._action_set[action]):  # Obstacle
            return self._get_obs(), -1, self.done(), {}
        # u2 = self._get_utility(None)
        u2 = self._get_utility(action)
        return self._get_obs(), u1 - u2, self.done(), {}

    def plan(self):
        self._sim.plan()
        actions = []
        for edge in self._sim._planner.iter_solution():
            i = np.argmin(np.abs(self._rotation_set - edge.get_odoms()[0].theta))
            actions.insert(0, i)
        return actions

    def done(self):
        return self._sim.step > self._max_steps

    def reset(self):
        while True:
            # Reset seed in configuration
            seed = self.np_random.randint(0, np.iinfo(np.int32).max, dtype=np.int32)
            self._config.set('Simulator', 'seed', str(seed))

            # Initialize new instance and perfrom a 360 degree scan of the surrounding
            self._sim = EMExplorer(self._config)
            for step in range(4):
                odom = 0, 0, math.pi / 2.0
                self._sim.simulate(odom)
            if self._sim._slam.map.get_landmark_size() < 1:
                continue

            # Return initial observation
            return self._get_obs()

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human':
            if self._viewer is None:
                self._sim.plot()
                self._viewer = plt.gcf()
                plt.ion()
                plt.tight_layout()
                plt.show()
            else:
                self._viewer.clf()
                self._sim.plot()
                plt.draw()
            plt.pause(ExplorationEnv.metadata['render.pause'])
        elif mode == 'state':
            assert(len(self._obs) == 3)
            if self._viewer is None:
                self._viewer = plt.subplots(1, 2, figsize=(12, 6))
                plot_virtual_map(self._sim._virtual_map, self._sim._map_params,
                                 alpha=1.0, ax=self._viewer[1][0])
                plot_virtual_map_cov(self._obs[1:], self._max_sigma,
                                     self._sim._map_params,
                                     alpha=1.0, ax=self._viewer[1][1])
                plt.ion()
                plt.tight_layout()
                plt.show()
            else:
                self._viewer[1][0].clear()
                plot_virtual_map(self._sim._virtual_map, self._sim._map_params,
                                 alpha=1.0, ax=self._viewer[1][0])
                self._viewer[1][1].clear()
                plot_virtual_map_cov(self._obs[1:], self._max_sigma,
                                     self._sim._map_params,
                                     alpha=1.0, ax=self._viewer[1][1])
                plt.draw()
            plt.pause(ExplorationEnv.metadata['render.pause'])
        else:
            super(ExplorationEnv, self).render(mode=mode)

if __name__ == '__main__':
    import sys
    config_file = sys.path[0] + '/../envs/exploration_env.ini'
    ExplorationEnv.metadata['render.pause'] = 0.10

    mode = 'human'
    env = ExplorationEnv(config_file, 50)
    env.render(mode=mode)
    for i in range(20):
        actions = env.plan()
        for a in actions[:5]:
            obs, reward, done, _ = env.step(a)
            print 'reward: ', reward, ', done: ', done
            env.render(mode=mode)
