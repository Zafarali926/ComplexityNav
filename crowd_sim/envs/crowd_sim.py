import logging
import random
import math

import gym
import matplotlib.lines as mlines
from matplotlib import patches
from matplotlib.collections import PatchCollection
from matplotlib import colors
import numpy as np
from numpy.linalg import norm

from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.state import tensor_to_joint_state, JointState
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.linear import Linear
from crowd_sim.envs.policy.socialforce import SocialForce


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.robot_sensor_range = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_scenario = None
        self.test_scenario = None
        self.current_scenario = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        self.nonstop_human = None
        self.centralized_planning = None
        self.centralized_planner = None

        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        self.robot_actions = None
        self.rewards = None
        self.As = None
        self.Xs = None
        self.feats = None
        self.trajs = list()
        self.panel_width = 10
        self.panel_height = 10
        self.panel_scale = 1
        self.test_scene_seeds = []
        self.dynamic_human_num = []
        self.human_starts = []
        self.human_goals = []

        self.phase = None

    def sample_uniform(self, range):
        return random.randint(range[0], range[1])

    def set_num_policies(self):
        config = self.config
        if config.humans.num_sf_orca is not None:
            num_orca = self.sample_uniform([0, config.humans.num_sf_orca])
            num_sf = config.humans.num_sf_orca - num_orca
        else:
            if len(config.humans.num_orca) > 1:
                num_orca = self.sample_uniform(config.humans.num_orca)
            else:
                num_orca = config.humans.num_orca[0]
            
            if len(config.humans.num_sf) > 1:
                num_sf = self.sample_uniform(config.humans.num_sf)
            else:
                num_sf = config.humans.num_sf[0]

        if config.humans.num_linear_static is not None:
            num_linear = self.sample_uniform([0, config.humans.num_linear_static])
            num_static = config.humans.num_linear_static - num_linear
        else:
            if len(config.humans.num_linear) > 1:
                num_linear = self.sample_uniform(config.humans.num_linear)
            else:
                num_linear = config.humans.num_linear[0]

            if len(config.humans.num_static) > 1:
                num_static = self.sample_uniform(config.humans.num_static)
            else:
                num_static = config.humans.num_static[0]

        self.num_policies = {
            'orca' : num_orca,
            'socialforce' : num_sf,
            'linear' : num_linear,
            'static' : num_static
        }


    def configure(self, config):
        self.config = config
        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.randomize_attributes = config.env.randomize_attributes
        self.robot_sensor_range = config.env.robot_sensor_range
        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty
        self.discomfort_dist = config.reward.discomfort_dist
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': config.env.train_size, 'val': config.env.val_size,
                          'test': config.env.test_size}
        self.train_val_scenario = config.sim.train_val_scenario
        self.test_scenario = config.sim.test_scenario
        self.square_width = config.sim.square_width
        self.circle_radius = config.sim.circle_radius
        self.human_num = config.sim.human_num
        self.random_seed = config.sim.random_seed

        self.nonstop_human = config.sim.nonstop_human
        self.centralized_planning = config.sim.centralized_planning
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        self.multi_policy = config.sim.multi_policy
        human_policy = config.humans.policy
        if self.centralized_planning:
            if human_policy == 'socialforce':
                logging.warning('Current socialforce policy only works in decentralized way with visible robot!')
            self.centralized_planner = policy_factory['centralized_' + human_policy]()

        #logging.info('human number: {}'.format(self.human_num))
        #if self.randomize_attributes:
        #    logging.info("Randomize human's radius and preferred speed")
        #else:
        #    logging.info("Not randomize human's radius and preferred speed")
        #logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_scenario, self.test_scenario))
        #logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))
        self.orca_border = self.create_border_orca(config.env.dx_range, config.env.dy_range)
        print("CONFIG DX DY RANGE: ", config.env.dx_range, config.env.dy_range)
        self.sfm_border = self.create_border_sfm(config.env.dx_range, config.env.dy_range)
        self.x_width = (config.env.dx_range[1] - config.env.dx_range[0]) - (2 * config.robot.radius + 1e-2)
        self.y_width = (config.env.dy_range[1] - config.env.dy_range[0]) - (2 * config.robot.radius + 1e-2)
        self.min_dist_sum = 0.0
        self.min_dist_overall = 1e6
        self.num_steps = 0
        self.robot_velocities = []
        self.robot_accelerations = []

        self.current_scenario = self.test_scenario
        #self.generate_scenarios_fixed()
        self.scenarios = None
        self.scenario_num = 0

        self.collided = False

    def set_robot(self, robot):
        self.robot = robot

    def create_border_orca(self, dx_range, dy_range):
        return [(dx_range[0], dy_range[1]), (dx_range[1], dy_range[1]), (dx_range[1], dy_range[0]), (dx_range[0], dy_range[0])]
    
    def create_border_sfm(self, dx_range, dy_range):
        lower = [dx_range[0], dx_range[1], dy_range[0] - 1, dy_range[0]]
        upper = [dx_range[0], dx_range[1], dy_range[1], dy_range[1] + 1]
        left = [dx_range[0] - 1, dx_range[0], dy_range[0], dy_range[1]]
        right = [dx_range[1], dx_range[1] + 1, dy_range[0], dy_range[1]]
        return [lower, upper, left, right]
    
    def min_dist_to_human(self):
        min_dist = 1e6
        robot_state = self.robot.get_full_state()
        for human in self.humans:
            state = human.get_full_state()
            dist = np.sqrt((robot_state.px - state.px)**2 + (robot_state.py - state.py)**2)
            if dist < min_dist:
                min_dist = dist
            
        return min_dist
    
    def set_params_crossing_fixed(self, radius, agents):
        if np.random.random() > 0.5:
                sign = -1
        else:
            sign = 1
        while True:
            #px = np.random.random() * self.square_width * 0.5 * sign
            #py = (np.random.random() - 0.5) * self.square_width
            px = np.random.random() * self.x_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.y_width
            collide = False
            for agent in agents:
                #print(agent)
                if norm((px - agent[0], py - agent[1])) < radius + 0.3 + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.x_width * 0.5 * - sign
            gy = (np.random.random() - 0.5) * self.y_width
            collide = False
            for agent in agents:
                if norm((gx - agent[2], gy - agent[3])) < radius + 0.3 + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        return px, py, gx, gy, 0, 0, 0

    def set_params_passing_fixed(self, radius, agents):
        if np.random.random() > 0.5:
                sign = -1
        else:
            sign = 1
        while True:
            #px = np.random.random() * self.square_width * 0.5 * sign
            #py = (np.random.random() - 0.5) * self.square_width
            py = np.random.random() * self.y_width * 0.5 * sign
            px = (np.random.random() - 0.5) * self.x_width
            collide = False
            for agent in agents:
                if norm((px - agent[0], py - agent[1])) < radius + 0.3 + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gy = np.random.random() * self.y_width * 0.5 * - sign
            if px > 0:
                gx = (np.random.random() * 0.5) * self.x_width
            else:
                gx = -1 * (np.random.random() * 0.5) * self.x_width
            #gx = (np.random.random() - 0.5) * self.x_width
            collide = False
            for agent in agents:
                if norm((gx - agent[2], gy - agent[3])) < radius + 0.3 + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        return px, py, gx, gy, 0, 0, 0
    
    def set_params_crossing(self, radius):
        if np.random.random() > 0.5:
                sign = -1
        else:
            sign = 1
        while True:
            #px = np.random.random() * self.square_width * 0.5 * sign
            #py = (np.random.random() - 0.5) * self.square_width
            px = np.random.random() * self.x_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.y_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.x_width * 0.5 * - sign
            gy = (np.random.random() - 0.5) * self.y_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        return px, py, gx, gy, 0, 0, 0

    def set_params_passing(self, radius):
        if np.random.random() > 0.5:
                sign = -1
        else:
            sign = 1
        while True:
            #px = np.random.random() * self.square_width * 0.5 * sign
            #py = (np.random.random() - 0.5) * self.square_width
            py = np.random.random() * self.y_width * 0.5 * sign
            px = (np.random.random() - 0.5) * self.x_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gy = np.random.random() * self.y_width * 0.5 * - sign
            if px > 0:
                gx = (np.random.random() * 0.5) * self.x_width
            else:
                gx = -1 * (np.random.random() * 0.5) * self.x_width
            #gx = (np.random.random() - 0.5) * self.x_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        return px, py, gx, gy, 0, 0, 0

    def generate_human(self, human=None, policy=None):
        print("CURRENT SCENARIO ", self.current_scenario)
        if human is None:
            if self.multi_policy:
                if policy == 'static':
                    human = Human(self.config, 'humans', policy=policy_factory['linear']())
                else:
                    human = Human(self.config, 'humans', policy=policy_factory[policy]())
                #human = Human(self.config, 'humans')
            else:
                human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()


        if self.current_scenario == 'circle_crossing':
            while True:
                angle = np.random.random() * np.pi/2
                # add some noise to simulate all the possible cases robot could meet with human
                px_noise = (np.random.random() - 0.5) * human.v_pref
                py_noise = (np.random.random() - 0.5) * human.v_pref
                px = self.circle_radius * np.cos(angle) + px_noise
                py = self.circle_radius * np.sin(angle) + py_noise
                if np.random.random() > 0.5:
                    px = px * -1

                if np.random.random() > 0.5:
                    py = py * -1

                collide = False
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if norm((px - agent.px, py - agent.py)) < min_dist or \
                            norm((px - agent.gx, py - agent.gy)) < min_dist:
                        collide = True
                        break
                if not collide:
                    break
            human.set(px, py, -px, -py, 0, 0, 0)

        elif self.current_scenario == 'passing':
            human.set(*self.set_params_passing(human.radius))

        elif self.current_scenario == 'crossing':
            human.set(*self.set_params_crossing(human.radius))

        elif self.current_scenario == 'passing_crossing':
            if np.random.random() > 0.5:
                sign = -1
            else:
                sign = 1

            if sign == 1:
                human.set(*self.set_params_passing(human.radius))
            else:
                human.set(*self.set_params_crossing(human.radius))

        elif self.current_scenario == 'random':
            while True:
                py = (np.random.random() - 0.5) * self.y_width
                px = (np.random.random() - 0.5) * self.x_width
                collide = False
                for agent in [self.robot] + self.humans:
                    if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            while True:
                gy = (np.random.random() - 0.5) * self.y_width
                gx = (np.random.random() - 0.5) * self.x_width
                collide = False
                for agent in [self.robot] + self.humans:
                    if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            human.set(px, py, gx, gy, 0, 0, 0)


        if policy == 'static':
            human.v_pref = 1e-2
            human.gx = human.px + 1e-2
            human.gy = human.py + 1e-2

        return human
    
    def generate_human_state(self, agents, policy=None):
        #print("CURRENT SCENARIO: ", self.current_scenario)
        params = None
        if self.current_scenario == 'circle_crossing':
            while True:
                angle = np.random.random() * np.pi/2
                # add some noise to simulate all the possible cases robot could meet with human
                px_noise = (np.random.random() - 0.5) * 1.0
                py_noise = (np.random.random() - 0.5) * 1.0
                px = self.circle_radius * np.cos(angle) + px_noise
                py = self.circle_radius * np.sin(angle) + py_noise
                if np.random.random() > 0.5:
                    px = px * -1

                if np.random.random() > 0.5:
                    py = py * -1

                collide = False
                for agent in agents:
                    min_dist = 0.3 + 0.3 + self.discomfort_dist
                    if norm((px - agent[0], py - agent[1])) < min_dist or \
                            norm((px - agent[2], py - agent[3])) < min_dist:
                        collide = True
                        break
                if not collide:
                    break
            params = px, py, -px, -py, 0, 0, 0

        elif self.current_scenario == 'passing':
            params = self.set_params_passing_fixed(0.3, agents)

        elif self.current_scenario == 'crossing':
            params = self.set_params_crossing_fixed(0.3, agents)

        elif self.current_scenario == 'passing_crossing':
            if np.random.random() > 0.5:
                sign = -1
            else:
                sign = 1

            if sign == 1:
                params = self.set_params_passing_fixed(0.3, agents)
            else:
                params = self.set_params_crossing_fixed(0.3, agents)
            #print("PARAMS: ", params)

        elif self.current_scenario == 'random':
            #print("IN RANDOM")
            while True:
                py = (np.random.random() - 0.5) * self.y_width
                px = (np.random.random() - 0.5) * self.x_width
                collide = False
                for agent in agents:
                    if norm((px - agent[0], py - agent[1])) < 0.3 + 0.3 + self.discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            while True:
                gy = (np.random.random() - 0.5) * self.y_width
                gx = (np.random.random() - 0.5) * self.x_width
                collide = False
                for agent in agents:
                    if norm((gx - agent[2], gy - agent[3])) < 0.3 + 0.3 + self.discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            params = px, py, gx, gy, 0, 0, 0
            #print("PARAMS ", params)

        if policy == 'static':
            params = list(params)
            params[2] = params[0] + 1e-2
            params[3] = params[1] + 1e-2
            params = tuple(params)
            #print(params)

        return params
    
    def generate_scenarios_fixed(self):
        states = []
        for i in range(500):
            states_i = {}
            agents = [(0, -4, 0, 4, 0, 0, np.pi / 2)]
            self.set_num_policies()
            for policy in self.num_policies:
                for _ in range(self.num_policies[policy]):
                    if policy in states_i:
                        params = self.generate_human_state(agents, policy=policy)
                        states_i[policy].append(params)
                        agents.append(params)
                    else:
                        states_i[policy] = [self.generate_human_state(agents, policy=policy)]
            states.append(states_i)

        self.scenarios = states
    
    def generate_human_from_state(self, policy, state):
        if self.multi_policy:
            if policy == 'static':
                human = Human(self.config, 'humans', policy=policy_factory['linear']())
            else:
                human = Human(self.config, 'humans', policy=policy_factory[policy]())
            #human = Human(self.config, 'humans')
        else:
            human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        print("SETTING STATE FOR POLICY: ", policy, state)
        human.set(*state)
        if policy == 'static':
            human.v_pref = 1e-4
        return human

    def reset(self, phase='test', scenario=None, test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        if self.robot is None:
            raise AttributeError('Robot has to be set!')

        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0

        base_seed = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                     'val': 0, 'test': self.case_capacity['val']}
        
        if self.robot.policy.name == 'vecmpc' or self.robot.policy.name == 'vecmppi':
            self.robot.policy.reset()

        self.collided = False

        self.min_dist_overall = 1e6
        self.robot_velocities = []

        self.robot.set(0, -4, 0, 4, 0, 0, np.pi / 2)
        if self.case_counter[phase] >= 0:
            seed = base_seed[phase] + self.case_counter[phase] + 0
            if self.random_seed:
                seed = random.randint(1000, 10000)
                #print("SEED RANDOM: ", seed)
            #print("SEED: ", seed)
            #np.random.seed(seed)
            #random.seed(seed)
            #print("SEED: ", seed)
            if phase == 'test':
                logging.debug('current test seed is:{}'.format(base_seed[phase] + self.case_counter[phase]))
            if not self.robot.policy.multiagent_training and phase in ['train', 'val']:
                # only CADRL trains in circle crossing simulation
                human_num = 1
                self.current_scenario = 'circle_crossing'
            else:
                self.current_scenario = self.test_scenario
                human_num = self.human_num
            self.humans = []
            if self.multi_policy:
                if scenario is not None:
                    print("YES THIS IS CORRECT", self.scenario_num)
                    #agent_states = scenario
                    for policy in scenario:
                        #print("NUM OF POLICY: ", policy, " ", len(scenario[policy]))
                        for n in range(len(scenario[policy])):
                            self.humans.append(self.generate_human_from_state(policy, scenario[policy][n]))
                else:
                    #print("WHY ARE WE IN HERE")
                    self.set_num_policies()
                    for policy in self.num_policies:
                        for _ in range(self.num_policies[policy]):
                            self.humans.append(self.generate_human(human=None, policy=policy))
            else:
                print("WE SHOULD NOT BE IN HERE")
                for _ in range(human_num):
                    self.humans.append(self.generate_human())

            #for human in self.humans:
                #print("HUMAN: ", human.px, human.py, human.gx, human.gy, human.policy, human.radius, human.v_pref)

            # case_counter is always between 0 and case_size[phase]
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        else:
            if self.random_seed:
                seed = random.randint(1000, 10000)
            #np.random.seed(seed)
            #random.seed(seed)
            print("IN ELSE SEED IS: ", seed)
            assert phase == 'test'
            if self.case_counter[phase] == -1:
                # for debugging purposes
                self.human_num = 3
                self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
            else:
                raise NotImplementedError
        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        if self.centralized_planning:
            self.centralized_planner.time_step = self.time_step

        self.states = list()
        self.robot_actions = list()
        self.rewards = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()
        if hasattr(self.robot.policy, 'get_matrix_A'):
            self.As = list()
        if hasattr(self.robot.policy, 'get_feat'):
            self.feats = list()
        if hasattr(self.robot.policy, 'get_X'):
            self.Xs = list()
        if hasattr(self.robot.policy, 'trajs'):
            self.trajs = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = self.compute_observation_for(self.robot)
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        return ob

    def onestep_lookahead(self, action):
        return self.step(action, update=False)
    
    def outside_check(self, agent_state, obstacle, action):
        px = agent_state.px + action.vx * self.time_step
        py = agent_state.py + action.vy * self.time_step

        left = px - agent_state.radius < obstacle[0][0]
        right = px + agent_state.radius > obstacle[1][0]
        if left or right:
            vx = 0.0
        else:
            vx = action.vx

        below = py - agent_state.radius < obstacle[2][1]
        above = py + agent_state.radius > obstacle[1][1]
        if below or above:
            vy = 0.0
        else:
            vy = action.vy

        return ActionXY(vx, vy)

    def step(self, action, update=True, baseline=None):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        #print("ACTION: ", action)
        #print("STEP")
        self.num_steps = self.num_steps + 1
        if self.centralized_planning:
            agent_states = [human.get_full_state() for human in self.humans]
            agent_policies = [human.policy for human in self.humans]
            #print("AGENT POLICIES: ", agent_policies)
            rstate = self.robot.get_full_state()
            #logging.info("PRE ACT ROBOT: px %.2f py %.2f vx %.2f vy %.2f v %.2f radius %.2f", rstate.px, rstate.py, rstate.vx, rstate.vy, np.sqrt(rstate.vx**2 + rstate.vy**2), rstate.radius)

            if self.robot.visible:
                agent_states.append(self.robot.get_full_state())
                if self.multi_policy:
                    human_actions, robot_actions = self.centralized_planner.predict(agent_states, agent_policies, self.orca_border, self.orca_border)
                    for human in self.humans:
                        hstate = human.get_full_state()
                        #logging.info("PRE ACT HUMAN: px %.2f py %.2f vx %.2f vy %.2f v %.2f radius %.2f", hstate.px, hstate.py, hstate.vx, hstate.vy, np.sqrt(hstate.vx**2 + hstate.vy**2), hstate.radius)
                    #print("CENTRALIZED ROBOT ACTION: ", robot_actions)
                    #print("HUMAN ACTIONS: ", human_actions)
                    #human_actions = self.centralized_planner.predict(agent_states)[:-1]
                else:
                    human_actions = self.centralized_planner.predict(agent_states)[:-1]
            else:
                if self.multi_policy:
                    #print("IN HERE!")
                    #human_actions = self.centralized_planner.predict(agent_states)
                    human_actions, robot_actions = self.centralized_planner.predict(agent_states, agent_policies, self.orca_border, self.orca_border)
                else:
                    human_actions = self.centralized_planner.predict(agent_states)
        else:
            #print("NOT CENTRALIZED")
            human_actions = []
            i = 0
            for human in self.humans:
                i = i + 1
                ob = self.compute_observation_for(human)
                human_actions.append(human.act(ob))

        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            #closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            closest_dist = np.sqrt(px**2 + py**2) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                logging.info("Collision: distance between robot and p{} is {:.2E} at time {:.2E}".format(human.id, closest_dist, self.global_time))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius
        self.min_dist_sum = self.min_dist_sum + self.min_dist_to_human()
        if self.min_dist_overall > self.min_dist_to_human():
            self.min_dist_overall = self.min_dist_to_human()

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            print("COLLISION")
            reward = self.collision_penalty
            done = True
            info = Collision()
            self.collided = True
        elif reaching_goal:
            reward = self.success_reward
            done = True
            print("MIN DISTANCE: ", self.min_dist_overall)
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Discomfort(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()

        if update:
            # store state, action value and attention weights
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())
            if hasattr(self.robot.policy, 'get_matrix_A'):
                self.As.append(self.robot.policy.get_matrix_A())
            if hasattr(self.robot.policy, 'get_feat'):
                self.feats.append(self.robot.policy.get_feat())
            if hasattr(self.robot.policy, 'get_X'):
                self.Xs.append(self.robot.policy.get_X())
            if hasattr(self.robot.policy, 'traj'):
                self.trajs.append(self.robot.policy.get_traj())

            # update all agents
            #print("ROBOT VX VY: ", self.robot.vx, self.robot.vy, " ACTION: ", action, len(self.robot_velocities))
            self.robot_velocities.append([self.robot.vx, self.robot.vy])
            if len(self.robot_velocities) > 2:
                vel1 = self.robot_velocities[len(self.robot_velocities) - 1]
                vel2 = self.robot_velocities[len(self.robot_velocities) - 2]
                ax = (vel1[0] - vel2[0]) / self.time_step
                ay = (vel1[1] - vel2[1]) / self.time_step
                #print("vel1 vel2 ax ay: ", vel1, vel2, ax, ay)
                #print(vel1, vel2, ax, ay, np.sqrt(ax**2 + ay**2), self.time_step, self.global_time)
                self.robot_accelerations.append(np.sqrt(ax**2 + ay**2))
            if baseline is not None:
                if baseline == 'orca':
                    #print("using orca action")
                    action = robot_actions[1]
                elif baseline == 'sfm':
                    action = robot_actions[0]
            action = self.outside_check(self.robot, self.orca_border, action)
            #print("ACTION RIGHT BEFORE STEP: ", action, "POSITION: ", self.robot.px, self.robot.py)
            self.robot.step(action)
            rstate = self.robot.get_full_state()
            #logging.info("POST ACT ROBOT: px %.2f py %.2f vx %.2f vy %.2f v %.2f radius %.2f", rstate.px, rstate.py, rstate.vx, rstate.vy, np.sqrt(rstate.vx**2 + rstate.vy**2), rstate.radius)
            #print("FINAL ROBOT ACTION: ", action)
            for human, action in zip(self.humans, human_actions):
                #print("HUMAN ACTION: ", action)
                action = self.outside_check(human, self.orca_border, action)
                human.step(action)

                hstate = human.get_full_state()
                #logging.info("POST ACT HUMAN: px %.2f py %.2f vx %.2f vy %.2f v %.2f radius %.2f", hstate.px, hstate.py, hstate.vx, hstate.vy, np.sqrt(hstate.vx**2 + hstate.vy**2), hstate.radius)

                if self.nonstop_human and human.reached_destination():
                    if human.v_pref > 1e-2:
                        self.generate_human(human)

            self.global_time += self.time_step
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans],
                                [human.id for human in self.humans]])
            self.robot_actions.append(action)
            self.rewards.append(reward)

            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob = self.compute_observation_for(self.robot)
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError

        print(ob, reward, done, info)
        return ob, reward, done, info

    def compute_observation_for(self, agent):
        if agent == self.robot:
            ob = []
            for human in self.humans:
                ob.append(human.get_observable_state())
        else:
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != agent]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
        return ob

    def render(self, mode='video', output_file='/home/socnav/arstr/rgl_density_1.mp4'):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        x_offset = 0.2
        y_offset = 0.4
        cmap = plt.cm.get_cmap('hsv', 3)
        robot_color = 'black'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)
        display_numbers = True

        print("WE ARE RENDERING NOW!")

        if mode == 'traj':
            print("TRAJ MODE ACTIVATE")
            fig, ax = plt.subplots(figsize=(7, 7))

            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add human start positions and goals
            #human_colors = [cmap(i) for i in range(len(self.humans))]
            human_colors = []
            for h in self.humans:
                print("LOOP")
                if isinstance(h.policy, SocialForce):
                    human_colors.append(cmap(0))
                elif isinstance(h.policy, ORCA):
                    human_colors.append(cmap(1))
                elif isinstance(h.policy, Linear):
                    print("LINEAR")
                    human_colors.append(cmap(2))

            for i in range(len(self.humans)):
                human = self.humans[i]
                human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                           color=human_colors[i],
                                           marker='*', linestyle='None', markersize=15)
                ax.add_artist(human_goal)
                human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                            color=human_colors[i],
                                            marker='o', linestyle='None', markersize=15)
                ax.add_artist(human_start)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]

            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=False, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)

                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                       ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
        elif mode == 'video':
            
            print("VIDEO MODE ACTIVATE")
            fig, ax = plt.subplots(figsize=(7, 7))
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
            ax.tick_params(bottom=False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            #ax.set_xlabel('x(m)', fontsize=14)
            #ax.set_ylabel('y(m)', fontsize=14)
            show_human_start_goal = False
            display_numbers = False

            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                 color=robot_color, marker='*', linestyle='None',
                                 markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            # sensor_range = plt.Circle(robot_positions[0], self.robot_sensor_range, fill=False, ls='dashed')
            ax.add_artist(robot)
            ax.add_artist(goal)

            print("ORCA BORDER: ", self.orca_border)
            for i in range(3):
                ax.plot([self.orca_border[i][0], self.orca_border[i+1][0]], [self.orca_border[i][1], self.orca_border[i+1][1]], color='black')
            ax.plot([self.orca_border[3][0], self.orca_border[0][0]], [self.orca_border[3][1], self.orca_border[0][1]], color='black')

            human_colors = []
            for h in self.humans:
                print(h.policy)
                if isinstance(h.policy, SocialForce):
                    human_colors.append('#4575b4')
                    #human_colors.append(0)
                elif isinstance(h.policy, ORCA):
                    human_colors.append('#1a9850')
                    #human_colors.append(0.2)
                elif isinstance(h.policy, Linear):
                    print("LINEAR")
                    human_colors.append('#998ec3')
                    #human_colors.append(0.4)

            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius)
                      for i in range(len(self.humans))]
            humans_patch = PatchCollection(humans, cmap=plt.cm.tab10, alpha=1.0)
            ax.add_collection(humans_patch)
            global_step = 0

            frames = len(self.states)
            if self.collided:
                frames = len(self.states) - 1
            collide = False

            def update(frame_num):
                nonlocal global_step
                nonlocal frames
                nonlocal collide
                global_step = frame_num
                #robot.center = robot_positions[frame_num]

                robot.center = robot_positions[frame_num]

                patches = []
                for i in range(len(human_positions[frame_num])):
                    #print("IN LOOOPP")
                    circle = plt.Circle(human_positions[frame_num][i], self.humans[i].radius)
                    patches.append(circle)

                humans_patch.set_paths(patches)
                print(colors.to_rgb(human_colors[0]))
                colors_list = []
                for i in range(len(human_colors)):
                    colors_list.append(colors.to_rgb(human_colors[i]))
                    if np.sqrt((robot.center[0] - human_positions[frame_num][i][0])**2 + (robot.center[1] - human_positions[frame_num][i][1])**2) < 0.6:
                        colors_list[i] = colors.to_rgb('#ff0000')
                        if not collide:
                            print("RAHHHHHH", frame_num)
                            frames = frame_num
                            collide = True
                        print("CCCCCCCCCCCCCCCCOOOOOOOOOOOOOOOOOOOOOOOOOOOOOLISIONNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN")
                color_arr = np.array(colors_list)
                #print("COLOR ARR: ", color_arr)
                #humans_patch.set_array(color_arr)
                humans_patch.set_facecolor(color_arr)

            print("FRAMES ", frames)
            anim = animation.FuncAnimation(fig, update, frames=frames, interval=self.time_step * 500, blit=False)

            '''
            for i in range(3):
                ax.plot([self.orca_border[i][0], self.orca_border[i+1][0]], [self.orca_border[i][1], self.orca_border[i+1][1]], color='black')
            ax.plot([self.orca_border[3][0], self.orca_border[0][0]], [self.orca_border[3][1], self.orca_border[0][1]], color='black')

            # add human start positions and goals
            #human_colors = [cmap(i) for i in range(len(self.humans))]
            human_colors = []
            for h in self.humans:
                print(h.policy)
                if isinstance(h.policy, SocialForce):
                    human_colors.append('#4575b4')
                elif isinstance(h.policy, ORCA):
                    human_colors.append('#1a9850')
                elif isinstance(h.policy, Linear):
                    print("LINEAR")
                    human_colors.append('#998ec3')
            if show_human_start_goal:
                print("WHY ARENT WE SEEING THEM?")
                for i in range(len(self.humans)):
                    human = self.humans[i]
                    human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                               color=human_colors[i],
                                               marker='*', linestyle='None', markersize=8)
                    ax.add_artist(human_goal)
                    human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                                color=human_colors[i],
                                                marker='o', linestyle='None', markersize=8)
                    ax.add_artist(human_start)
            # add robot start position
            robot_start = mlines.Line2D([self.robot.get_start_position()[0]], [self.robot.get_start_position()[1]],
                                        color=robot_color,
                                        marker='o', linestyle='None', markersize=8)
            robot_start_position = [self.robot.get_start_position()[0], self.robot.get_start_position()[1]]
            ax.add_artist(robot_start)
            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                 color=robot_color, marker='*', linestyle='None',
                                 markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            # sensor_range = plt.Circle(robot_positions[0], self.robot_sensor_range, fill=False, ls='dashed')
            ax.add_artist(robot)
            ax.add_artist(goal)
            #plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=14)

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=True, color=human_colors[i])
                      for i in range(len(self.humans))]
            humans_patch = PatchCollection(humans, animated=True)
            ax.add_collection(humans_patch)

            # disable showing human numbers
            if display_numbers:
                human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset, str(i),
                                          color='black') for i in range(len(self.humans))]

            for i, human in enumerate(humans):
                #ax.add_artist(human)
                if display_numbers:
                    ax.add_artist(human_numbers[i])

            # add time annotation
            #time = plt.text(0.4, 1.05, 'Time: {}'.format(0), fontsize=16, transform=ax.transAxes)
            #ax.add_artist(time)

            # visualize attention scores
            # if hasattr(self.robot.policy, 'get_attention_weights'):
            #     attention_scores = [
            #         plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
            #                  fontsize=16) for i in range(len(self.humans))]

            # compute orientation in each step and use arrow to show the direction
            
            # radius = self.robot.radius
            # orientations = []
            # for i in range(self.human_num + 1):
            #     orientation = []
            #     for state in self.states:
            #         agent_state = state[0] if i == 0 else state[1][i - 1]
            #         if self.robot.kinematics == 'unicycle' and i == 0:
            #             direction = (
            #             (agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(agent_state.theta),
            #                                                agent_state.py + radius * np.sin(agent_state.theta)))
            #         else:
            #             theta = np.arctan2(agent_state.vy, agent_state.vx)
            #             direction = ((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
            #                                                             agent_state.py + radius * np.sin(theta)))
            #         orientation.append(direction)
            #     orientations.append(orientation)
            #     if i == 0:
            #         arrow_color = 'black'
            #         arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)]
            #     else:
            #         arrows.extend(
            #             [patches.FancyArrowPatch(*orientation[0], color=human_colors[i - 1], arrowstyle=arrow_style)])

            # for arrow in arrows:
            #     ax.add_artist(arrow)
            global_step = 0

            # if len(self.trajs) != 0:
            #     human_future_positions = []
            #     human_future_circles = []
            #     for traj in self.trajs:
            #         human_future_position = [[tensor_to_joint_state(traj[step+1][0]).human_states[i].position
            #                                   for step in range(self.robot.policy.planning_depth)]
            #                                  for i in range(self.human_num)]
            #         human_future_positions.append(human_future_position)

            #     for i in range(self.human_num):
            #         circles = []
            #         for j in range(self.robot.policy.planning_depth):
            #             circle = plt.Circle(human_future_positions[0][i][j], self.humans[0].radius/(1.7+j), fill=False, color=cmap(i))
            #             ax.add_artist(circle)
            #             circles.append(circle)
            #         human_future_circles.append(circles)

            def update(frame_num):
                nonlocal global_step
                #nonlocal arrows
                global_step = frame_num
                robot.center = robot_positions[frame_num]

                patches = []
                # for i, human in enumerate(humans):
                #     human.center = human_positions[frame_num][i]
                #     if np.sqrt((robot.center[0] - human.center[0])**2 + (robot.center[1] - human.center[1])**2) < 0.6:
                #         print("ROBOT CENTER: ", robot.center, " HUMAN CENTER: ", human.center)
                #         #human.set_color('red')
                #         human_colors[i] = '#ff0000'
                #         circ = plt.Circle(human_positions[frame_num][i], self.humans[i].radius, fill=True, color='#ff0000')
                #         ax.add_patch(circ)
                #     if display_numbers:
                #         human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] + y_offset))
                # color_arr = np.array(colors.to_rgb(c) for c in human_colors)
                # humans_patch.set_array(color_arr)

                for i in range(len(human_positions[frame_num])):
                    circle = plt.Circle(human_positions[frame_num][i], self.humans[i].radius, fill=True, color=human_colors[i])
                    patches.append(circle)

                humans_patch.set_paths(patches)
                color_arr = np.array(colors.to_rgb(c) for c in human_colors)
                humans_patch.set_array(color_arr)

                return humans_patch

                # for arrow in arrows:
                #     arrow.remove()

                # for i in range(self.human_num + 1):
                #     orientation = orientations[i]
                #     if i == 0:
                #         arrows = [patches.FancyArrowPatch(*orientation[frame_num], color='black',
                #                                           arrowstyle=arrow_style)]
                #     else:
                #         arrows.extend([patches.FancyArrowPatch(*orientation[frame_num], color=cmap(i - 1),
                #                                                arrowstyle=arrow_style)])

                # for arrow in arrows:
                #     ax.add_artist(arrow)
                #     # if hasattr(self.robot.policy, 'get_attention_weights'):
                #     #     attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                #time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

                # if len(self.trajs) != 0:
                #     for i, circles in enumerate(human_future_circles):
                #         for j, circle in enumerate(circles):
                #             circle.center = human_future_positions[global_step][i][j]

            def plot_value_heatmap():
                if self.robot.kinematics != 'holonomic':
                    print('Kinematics is not holonomic')
                    return
                # for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                #     print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                #                                              agent.vx, agent.vy, agent.theta))

                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (self.robot.policy.rotation_samples, self.robot.policy.speed_samples))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def print_matrix_A():
                # with np.printoptions(precision=3, suppress=True):
                #     print(self.As[global_step])
                h, w = self.As[global_step].shape
                print('   ' + ' '.join(['{:>5}'.format(i - 1) for i in range(w)]))
                for i in range(h):
                    print('{:<3}'.format(i-1) + ' '.join(['{:.3f}'.format(self.As[global_step][i][j]) for j in range(w)]))
                # with np.printoptions(precision=3, suppress=True):
                #     print('A is: ')
                #     print(self.As[global_step])

            def print_feat():
                with np.printoptions(precision=3, suppress=True):
                    print('feat is: ')
                    print(self.feats[global_step])

            def print_X():
                with np.printoptions(precision=3, suppress=True):
                    print('X is: ')
                    print(self.Xs[global_step])

            def on_click(event):
                if anim.running:
                    anim.event_source.stop()
                    if event.key == 'a':
                        if hasattr(self.robot.policy, 'get_matrix_A'):
                            print_matrix_A()
                        if hasattr(self.robot.policy, 'get_feat'):
                            print_feat()
                        if hasattr(self.robot.policy, 'get_X'):
                            print_X()
                        # if hasattr(self.robot.policy, 'action_values'):
                        #    plot_value_heatmap()
                else:
                    anim.event_source.start()
                anim.running ^= True

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 500, blit=False)
            anim.running = True

            print("OUTPUT FILE: ", output_file)
            '''

            if output_file is not None:
                # save as video
                ffmpeg_writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                # writer = ffmpeg_writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=ffmpeg_writer)

                # save output file as gif if imagemagic is installed
                # anim.save(output_file, writer='imagemagic', fps=12)
            plt.show()
        else:
            raise NotImplementedError
