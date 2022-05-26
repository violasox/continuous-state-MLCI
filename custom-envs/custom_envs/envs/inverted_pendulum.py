import akro
import gym
import numpy as np
import pickle
import pybullet

from garage import EnvSpec
from gym import spaces
# from pybullet_envs.robot_pendula import InvertedPendulum
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from custom_envs.robot_bases import MJCFBasedRobot
# from pybullet_envs.robot_bases import MJCFBasedRobot
from pybullet_envs.scene_abstract import SingleRobotEmptyScene

from custom_envs import EXP5_FILE

class Cartpole(MJCFBasedRobot):
    def __init__(self):
        MJCFBasedRobot.__init__(self, 'inverted_pendulum.xml', 'cart', action_dim=1, obs_dim=5)

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        self.pole = self.parts['pole']
        self.slider = self.jdict['custom_slider']
        self.j1 = self.jdict['hinge']
        u = self.np_random.uniform(low=-.1, high=.1)
        self.j1.reset_current_position(u, 0)
        self.j1.set_motor_torque(0)
        numLinks = self._p.getNumJoints(0)
        for i in range(numLinks):
            #pybullet.changeVisualShape(self.objects, i, rgbaColor=[0.8, 0.1, 0.1, 1])
            pass

    def apply_action(self, a):
        assert(np.isfinite(a).all())
        self.slider.set_motor_torque(100 * float(np.clip(a[0], -1, +1)))

    def calc_state(self):
        (self.theta, theta_dot) = self.j1.current_position()
        (x, x_dot) = self.slider.current_position()
        
        if not np.isfinite(x):
          print("x is inf")
          x = 0

        if not np.isfinite(x_dot):
          print("c_dot is inf")
          x_dot = 0

        if not np.isfinite(self.theta):
          print("theta is inf")
          self.theta = 0

        if not np.isfinite(theta_dot):
          print("theta_dot is inf")
          theta_dot = 0

        return np.array([x, x_dot, np.cos(self.theta), np.sin(self.theta), theta_dot])

class InvertedPendulum(MJCFBasedRobot):
    def __init__(self, obs_dim=3):
        MJCFBasedRobot.__init__(self, 'non_cartpole_pendulum.xml', 'cart', action_dim=1, obs_dim=obs_dim)
        self.fixed_start_vec = None
        self.start_vec = None
        self.fixed_goal_vec = None
        self.goal_vec = None

    def robot_specific_reset(self, bullet_client):
        self.pivot = self.jdict['pivot']
        if self.fixed_start_vec is None:
            theta = self.np_random.uniform(low=0, high=2*np.pi)
            theta_dot = self.np_random.uniform(low=-2, high=2)
            self.start_vec = np.array([theta, theta_dot])
        else:
            self.start_vec = self.fixed_start_vec
        self.pivot.reset_current_position(self.start_vec[0], self.start_vec[1])
        self.pivot.set_motor_torque(0)

    def apply_action(self, a):
        assert(np.isfinite(a).all())
        self.pivot.set_motor_torque(10 * float(np.clip(a[0], -1, +1)))

    def calc_state(self):
        (self.theta, theta_dot) = self.pivot.current_position()

        if not np.isfinite(self.theta):
          print("theta is inf")
          self.theta = 0

        if not np.isfinite(theta_dot):
          print("theta_dot is inf")
          theta_dot = 0

        return np.array([np.cos(self.theta), np.sin(self.theta), theta_dot])

class InvertedPendulumWithGoal(InvertedPendulum):
    def __init__(self, obs_dim=6):
        super().__init__(obs_dim=obs_dim)

    def calc_state(self):
        state = super().calc_state()
        theta_vec = state[0:2]
        goal_theta_vec = np.array([np.cos(self.goal_vec[0]), np.sin(self.goal_vec[0])])
        # Use distance to goal for the observation
        theta_dist_vec = goal_theta_vec - theta_vec
        augmented_theta_dot = np.array([state[2], self.goal_vec[1] - state[2]])
        augmented_state = np.concatenate((theta_vec, theta_dist_vec, augmented_theta_dot))
        return augmented_state

    def robot_specific_reset(self, bullet_client):
        # Initialize at starting position
        super().robot_specific_reset(bullet_client)
        # Set goal position
        if self.fixed_goal_vec is None:
            # Pick a random goal while making sure robot doesn't start at the goal state
            good_goal = False
            while not good_goal:
                goal_theta = self.np_random.uniform(low=0, high=2*np.pi)
                goal_theta_dot = self.np_random.uniform(low=-2, high=2)
                self.goal_vec = np.array([goal_theta, goal_theta_dot])
                start_theta = self.start_vec[0]
                theta_dist = np.arccos(np.dot([np.cos(start_theta), np.sin(start_theta)], 
                                              [np.cos(goal_theta), np.sin(goal_theta)]))
                theta_dot_dist = np.abs(goal_theta_dot - self.start_vec[1])
                if max(theta_dist, theta_dot_dist) > 0.5:
                    good_goal = True
        else:
            self.goal_vec = self.fixed_goal_vec

class InvertedPendulumFiniteHorizon(InvertedPendulumWithGoal):
    def __init__(self):
        super().__init__(obs_dim=7)
        self.step_count = None
        self.delta = 0.005
        with open(EXP5_FILE, 'rb') as f:
            self.exp_dict = pickle.load(f)
        self.num_exp = self.exp_dict['numTrials']

    def robot_specific_reset(self, bullet_client):
        # exp = np.random.randint(low=0, high=self.num_exp)
        exp = 0
        # self.fixed_start_vec = self.exp_dict['startPoints'][:,exp]
        # self.fixed_goal_vec = self.exp_dict['goalPoints'][:,exp]
        super().robot_specific_reset(bullet_client)
        self.step_count = 0

    def calc_state(self):
        augmented_state = np.zeros(7)
        augmented_state[0:6] = super().calc_state()
        augmented_state[6] = self.step_count*self.delta
        return augmented_state

class CustomInvertedPendulumEnv(MJCFBaseBulletEnv):
    def __init__(self, robotClass=InvertedPendulum):
        self.robot = robotClass()
        MJCFBaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1
        # self.robot.fixed_start_vec = np.array([2*np.pi - 0.3, 0])

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0165, frame_skip=1)

    def reset(self):
        # TODO what does this do?
        if self.stateId >= 0:
            self._p.restoreState(self.stateId)
        resetEnv = MJCFBaseBulletEnv.reset(self)
        if self.stateId < 0:
            self.stateId = self._p.saveState()
        return resetEnv

    def get_episode_info(self):
        episode_info = {'fixed_start_vec': self.robot.fixed_start_vec,
                        'fixed_goal_vec': self.robot.fixed_goal_vec,
                        'start_vec': self.robot.start_vec,
                        'goal_vec': self.robot.goal_vec,
        }
        return episode_info

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()
        reward = np.cos(self.robot.theta)
        done = False
        self.rewards = [float(reward)]
        self.HUD(state, a, done)
        # Reward was originally sum(self.rewards) for some reason
        return (state, reward, done, {})

    # TODO ?
    def camera_adjust(self):
        self.camera.move_and_look_at(0, 1.0, 1.0, 0, 0, 0.5)

class CustomCartpoleEnv(CustomInvertedPendulumEnv):
    def __init__(self):
        super().__init__(robotClass=Cartpole)

class InvertedPendulumWithGoalEnv(CustomInvertedPendulumEnv):
    def __init__(self, robotClass=InvertedPendulumWithGoal):
        super().__init__(robotClass=robotClass)

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        augmented_state = self.robot.calc_state()
        current_theta = augmented_state[0:2]
        goal_theta = np.array([np.cos(self.robot.goal_vec[0]), np.sin(self.robot.goal_vec[0])])
        # arccos has range [0, pi] (always positive)
        theta_dist = np.arccos(np.dot(current_theta, goal_theta))
        theta_dot_dist = np.abs(augmented_state[4] - self.robot.goal_vec[1])
        if max(theta_dist, theta_dot_dist) < 0.5:
            done = True
            reward = 10
        else:
            done = False
            reward = -1*np.abs(a)**2*.0165
        self.rewards = [float(reward)]
        self.HUD(augmented_state, a, done)
        return (augmented_state, reward, done, {})

    def set_fixed_start(self, start_vec):
        self.robot.fixed_start_vec = start_vec

    def reset_fixed_start(self):
        self.robot.fixed_start_vec = None

    def set_fixed_goal(self, goal_vec):
        self.robot.fixed_goal_vec = goal_vec

    def reset_fixed_goal(self):
        self.robot.fixed_goal_vec = None

class InvertedPendulumFiniteHorizonEnv(InvertedPendulumWithGoalEnv):
    def __init__(self):
        super().__init__(robotClass=InvertedPendulumFiniteHorizon)
        self.delta = .0165
        self.time_horizon = int(5/self.delta)
    
    def step(self, a):
        self.robot.step_count += 1
        self.robot.apply_action(a)
        self.scene.global_step()
        augmented_state = self.robot.calc_state()
        current_theta = augmented_state[0:2]
        goal_theta = np.array([np.cos(self.robot.goal_vec[0]), np.sin(self.robot.goal_vec[0])])
        # arccos has range [0, pi] (always positive)
        theta_dist = np.arccos(np.dot(current_theta, goal_theta))
        theta_dot_dist = np.abs(augmented_state[4] - self.robot.goal_vec[1])
        if self.robot.step_count == self.time_horizon-1:
            done = True
            reward = -80*(theta_dist**2 + theta_dot_dist**2)
        else:
            done = False
            reward = -0.05 * (10*a[0])**2 * self.delta
        self.rewards = [float(reward)]
        self.HUD(augmented_state, a, done)
        return (augmented_state, reward, done, {})

class InvertedPendulumDiscreteActionEnv(InvertedPendulumFiniteHorizonEnv):
    def __init__(self):
        super().__init__()
        self._action_space = akro.Discrete(5)
        self._action_space = spaces.Discrete(5)
        self.action_space = spaces.Discrete(5)
        self.action_vals = [-1, -0.5, 0, 0.5, 1]
#         self._spec = EnvSpec(action_space=self._action_space, 
#                              observation_space = self._observation_space,
#                              max_episode_length=self.max_episode_length)

#     @property
#     def action_space(self):
#         return self._action_space
# 
#     @property
#     def spec(self):
#         return self._spec  

    def step(self, a):
        action = [self.action_vals[a]]
        return super().step(action)
