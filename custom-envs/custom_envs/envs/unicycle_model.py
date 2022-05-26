import akro
import gym
import numpy as np
from scipy.integrate import solve_ivp

from garage import Environment, EnvSpec, EnvStep, StepType

class UnicycleRacetrackEnv(Environment):
    def __init__(self):
        self.fixed_start_state = None
        self.start_state = None
        self.fixed_goal_state = np.array([0, 3, 0, 0])
        self.goal_state = None
        self.goal_mask = np.array([1, 1, 0, 0])
        self.delta = 0.0165
        self.goal_thresh = 0.5
        self.horizon = 3
        self.max_episode_length = int(self.horizon/self.delta)
        self._step_count = None
        self.out_of_bounds_penalty = 100
        self.num_actions = 3
        self.u1_vals = np.linspace(-10, 10, self.num_actions)
        self.u2_vals = np.linspace(-10, 10, self.num_actions)

    @property
    def observation_space(self):
        return akro.Box(low=-np.inf, high=np.inf, shape=(6,))

    @property
    def action_space(self):
        return akro.Discrete(self.num_actions**2)

    @property
    def spec(self):
        return EnvSpec(action_space=self.action_space, observation_space=self.observation_space, max_episode_length=self.max_episode_length)
        # return EnvSpec(action_space=self.action_space, observation_space=self.observation_space)
    
    @property
    def render_modes(self):
        return ['rgb_array']

    def reset(self):
        # Starting state
        if self.fixed_start_state is None:
            # heading = np.random.uniform(low=-np.pi/2, high=np.pi/2)
            # heading = np.random.uniform(low=0, high=2*np.pi)
            heading = np.pi/2
            speed = np.random.uniform(low=0, high=4)
            # x_pos = np.random.uniform(low=-1, high=1)
            x_pos = 0
            y_pos = np.random.uniform(low=-3.5, high=-1.5)
            self.start_state = np.array([x_pos, y_pos, heading, speed])
        else:
            self.start_state = self.fixed_start_state
        self._state = self.start_state.copy()
        self._step_count = 0
        
        # Goal state
        if self.fixed_goal_state is None:
            # Pick a random goal while making sure robot doesn't start at the goal state
            # TODO placeholder code from pendulum, this will break!
            good_goal = False
            while not good_goal:
                goal_theta = np.random.uniform(low=0, high=2*np.pi)
                goal_theta_dot = np.random.uniform(low=-2, high=2)
                self.goal_state = np.array([goal_theta, goal_theta_dot])
                start_theta = self.start_state[0]
                theta_dist = np.arccos(np.dot([np.cos(start_theta), np.sin(start_theta)], 
                                              [np.cos(goal_theta), np.sin(goal_theta)]))
                theta_dot_dist = np.abs(goal_theta_dot - self.start_state[1])
                if max(theta_dist, theta_dot_dist) > self.goal_thresh:
                    good_goal = True
        else:
            self.goal_state = self.fixed_goal_state.copy()
        
        observation = self.get_observation()
        self._step_count = 0
        
        return (observation, self.get_episode_info())

    def get_observation(self):
        obs = np.zeros(6)
        obs[0] = self._state[0]
        obs[1] = self._state[1]
        obs[2] = np.cos(self._state[2])
        obs[3] = np.sin(self._state[2])
        obs[4] = self._state[3]
        obs[5] = self._step_count
        return obs

    def step(self, action_index):
        self._step_count += 1
        action_indices = np.unravel_index(action_index, (self.num_actions, self.num_actions))
        u1 = self.u1_vals[action_indices[0]]
        u2 = self.u2_vals[action_indices[1]]
        sol = solve_ivp(unicycle_dynamics, [0, self.delta], self._state, args=(u1,u2))
        self._state = sol.y[:,-1]
        obs = self.get_observation()
        
        if self._step_count == self.max_episode_length:
            x_dist = np.abs(self._state[0] - self.goal_state[0])
            y_dist = np.abs(self._state[1] - self.goal_state[1])
            goal_head = self.goal_state[2]
            heading_dist = np.arccos(np.dot(obs[2:4], [np.cos(goal_head), np.sin(goal_head)]))
            speed_dist = np.abs(self._state[3] - self.goal_state[3])
            reward = -40*((self.goal_mask[0]*x_dist)**2 + (self.goal_mask[1]*y_dist))**2
            done = False
        else:
            done = False
            reward = -(u1**2 + u2**2)*self.delta
        step_type = StepType.get_step_type(step_cnt=self._step_count, max_episode_length = self.max_episode_length, 
                                           done=done)
        return EnvStep(env_spec=self.spec, action=action_index, reward=reward, observation=obs, step_type=step_type, 
                       env_info={'success': done, 'u1': u1, 'u2': u2})

    def render(self, mode):
        if mode=='ascii':
            return str(self._state)
        elif mode=='rgb_array':
            im_arr = np.ones((240, 320, 3), dtype='uint8')*255 # white
            center_x = 159
            center_y = 119
            radius = 80
            s = 3
            theta = -(self._state[0] - np.pi/2)
            pend_x = center_x + int(radius*np.cos(theta))
            pend_y = center_y - int(radius*np.sin(theta))
            goal_theta = -(self.goal_state[0] - np.pi/2)
            goal_x = center_x + int(radius*np.cos(goal_theta))
            goal_y = center_y - int(radius*np.sin(goal_theta))
            im_arr[center_y-s:center_y+s+1, center_x-s:center_x+s+1, :] = 0 # black
            im_arr[goal_y-s:goal_y+s+1, goal_x-s:goal_x+s+1, 1:3] = 0 # some other color
            im_arr[pend_y-s:pend_y+s+1, pend_x-s:pend_x+s+1, 1] = 0 # some color
            return im_arr

    def visualize(self):
        print(self.render('ascii'))

    def close(self):
        pass

    def set_fixed_start(self, start_state):
        self.fixed_start_state = start_state

    def set_fixed_goal(self, goal_state):
        self.fixed_goal_state = goal_state
    
    def get_episode_info(self):
        episode_info = {'fixed_start_state': self.fixed_start_state,
                        'fixed_goal_state': self.fixed_goal_state,
                        'start_state': self.start_state,
                        'goal_state': self.goal_state,
        }
        return episode_info

class StayInStraightLaneEnv(UnicycleRacetrackEnv):
    def __init__(self):
        super().__init__()
        self.fixed_goal_state = np.array([0.5, 5, 0, 0])
        self.max_episode_length=1000
    
    @property
    def observation_space(self):
        return akro.Box(low=-np.inf, high=np.inf, shape=(5,))

    def reset(self):
        self.goal_state = self.fixed_goal_state
        if self.fixed_start_state is None:
            x_pos = np.random.uniform(low=-0.8, high=0.8)
            # x_pos = 0.5
            y_pos = np.random.uniform(low=0, high=4.5)
            heading = np.pi/2 + np.random.uniform(low=-0.2, high=0.2)
            speed = 0
            self.start_state = np.array([x_pos, y_pos, heading, speed])
        else:
            self.start_state = self.fixed_start_state
        self._state = self.start_state
        self._step_count = 0
        observation = self.get_observation()[0:5]
        return (observation, self.get_episode_info())

    def step(self, action_index):
        self._step_count += 1
        action_indices = np.unravel_index(action_index, (self.num_actions, self.num_actions))
        u1 = self.u1_vals[action_indices[0]]
        u2 = self.u2_vals[action_indices[1]]
        sol = solve_ivp(unicycle_dynamics, [0, self.delta], self._state, args=(u1,u2))
        self._state = sol.y[:,-1]
        obs = self.get_observation()[0:5]

        if self._state[1] >= self.goal_state[1]:
            done = True
            reward = 10
        else:
            done = False
            lane_center_dist = np.abs(self._state[0] - 0.5)
            goal_dist = np.abs(self._state[1] - self.goal_state[1]) 
            reward = -self.delta*(5*lane_center_dist**2 + goal_dist**2)
        
        step_type = StepType.get_step_type(step_cnt=self._step_count, max_episode_length=self.max_episode_length, 
                                           done=done)
        return EnvStep(env_spec=self.spec, action=action_index, reward=reward, observation=obs, step_type=step_type, 
                       env_info={'success': done, 'u1': u1, 'u2': u2})
    

def unicycle_dynamics(t, y, u1, u2):
    x0_dot = y[3] * np.cos(y[2]) # x position
    x1_dot = y[3] * np.sin(y[2]) # y position
    x2_dot = u1 # turning velocity
    x3_dot = u2 # acceleration
    return [x0_dot, x1_dot, x2_dot, x3_dot]

class RelativeToLaneEnv(Environment):
    def __init__(self):
        self.fixed_start_state = None
        self.start_state = None
        self.fixed_goal_state = np.array([0, 8.5, 0, 0])
        self.goal_state = None
        self.delta = 0.05
        self.horizon = 10
        # self.max_episode_length = int(self.horizon/self.delta)
        self.max_episode_length=500
        self._step_count = None
        self.num_actions = 3
        self.u1_vals = np.linspace(-0.5, 0.5, self.num_actions)
        self.u2_vals = np.linspace(-1, 1, self.num_actions)
        self.crashed = False

    @property
    def observation_space(self):
        return akro.Box(low=-np.inf, high=np.inf, shape=(4,))

    @property
    def action_space(self):
        return akro.Discrete(self.num_actions**2)

    @property
    def spec(self):
        return EnvSpec(action_space=self.action_space, observation_space=self.observation_space, max_episode_length=self.max_episode_length)
    
    @property
    def render_modes(self):
        return ['rgb_array']

    def reset(self):
        # Starting state
        if self.fixed_start_state is None:
            heading = np.random.uniform(low=-0.2, high=0.2)
            #speed = np.random.uniform(low=0, high=4)
            speed = 1
            x_pos = np.random.uniform(low=-0.6, high=0)
            y_pos = np.random.uniform(low=-1, high=1)
            self.start_state = np.array([x_pos, y_pos, heading, speed])
        else:
            self.start_state = self.fixed_start_state
        self._state = self.start_state.copy()
        self._step_count = 0
        self.crashed = False
        
        self.goal_state = self.fixed_goal_state.copy()
        
        observation = self._state.copy()
        self._step_count = 0
        
        return (observation, self.get_episode_info())

    def step(self, action_index):
        self._step_count += 1
        action_indices = np.unravel_index(action_index, (self.num_actions, self.num_actions))
        u1 = self.u1_vals[action_indices[0]]
        u2 = self.u2_vals[action_indices[1]]
        sol = solve_ivp(relative_unicycle_dynamics, [0, self.delta], self._state, args=(u1,u2))
        self._state = sol.y[:,-1]
        obs = self._state.copy()
        
        x_dist = np.abs(self._state[0] - self.goal_state[0])
        y_dist = np.abs(self._state[1] - self.goal_state[1])
        # finished
        if self._state[1] > self.goal_state[1]:
            done = True
            reward = 100
        # out of bounds
        elif self._state[0] < -0.8 or self._state[0] > 0.2:
            self.crashed = True
            done = True
            reward = -100
        else:
            #if self._state[0] < -0.8 or self._state[0] > 0.2:
            #    self.crashed = True
            done = False
            reward = self.delta*(-((x_dist/0.01) + (y_dist/20)))
        step_type = StepType.get_step_type(step_cnt=self._step_count, max_episode_length = self.max_episode_length, 
                                           done=done)
        return EnvStep(env_spec=self.spec, action=action_index, reward=reward, observation=obs, step_type=step_type, 
                       env_info={'success': done and not self.crashed, 'u1': u1, 'u2': u2})

    def render(self, mode):
        if mode=='ascii':
            return str(self._state)
        elif mode=='rgb_array':
            pass

    def visualize(self):
        print(self.render('ascii'))

    def close(self):
        pass

    def set_fixed_start(self, start_state):
        self.fixed_start_state = start_state

    def set_fixed_goal(self, goal_state):
        self.fixed_goal_state = goal_state
    
    def get_episode_info(self):
        episode_info = {'fixed_start_state': self.fixed_start_state,
                        'fixed_goal_state': self.fixed_goal_state,
                        'start_state': self.start_state,
                        'goal_state': self.goal_state,
        }
        return episode_info

def relative_unicycle_dynamics(t, y, u1, u2):
    x0_dot = y[3] * np.sin(y[2]) # x position
    x1_dot = y[3] * np.cos(y[2]) # y position
    x2_dot = u1 # turning velocity
    x3_dot = u2 # acceleration
    return [x0_dot, x1_dot, x2_dot, x3_dot]
