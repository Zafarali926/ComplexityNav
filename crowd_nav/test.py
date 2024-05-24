import logging
import argparse
import importlib.util
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
import random
import time
from numpy.linalg import norm

def main_experiments(args):

    #np.random.seed(1000)
    #random.seed(1000)

    #ec = config.ExperimentsConfig(args.debug)
    #scenarios = random_sequence(ec)
    #scenarios = None

    # configure logging and device
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    if args.model_dir is not None:
        if args.config is not None:
            config_file = args.config
        else:
            config_file = os.path.join(args.model_dir, 'config.py')
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
            logging.info('Loaded IL weights')
        elif args.rl:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                print(os.listdir(args.model_dir))
                model_weights = os.path.join(args.model_dir, sorted(os.listdir(args.model_dir))[-1])
            logging.info('Loaded RL weights')
        else:
            model_weights = os.path.join(args.model_dir, 'best_val.pth')
            logging.info('Loaded RL weights with best VAL')

    else:
        config_file = args.config

    spec = importlib.util.spec_from_file_location('config', config_file)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    ec = config.ExperimentsConfig(args.debug)
    scenarios = random_sequence(ec)

    # configure policy
    policy_config = config.PolicyConfig(args.debug)
    if args.policy == 'vecmpc' or args.policy == 'vecmppi':
        env_config = config.EnvConfig(args.debug)
        policy = policy_factory[args.policy](env_config)
    else:
        policy = policy_factory[policy_config.name]()

    if args.planning_depth is not None:
        policy_config.model_predictive_rl.do_action_clip = True
        policy_config.model_predictive_rl.planning_depth = args.planning_depth
    if args.planning_width is not None:
        policy_config.model_predictive_rl.do_action_clip = True
        policy_config.model_predictive_rl.planning_width = args.planning_width
    if args.sparse_search:
        policy_config.model_predictive_rl.sparse_search = True

    print("POLICY: ", policy, "CONFIG: ", policy_config)

    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.load_model(model_weights)

    if args.visualize:
        # configure environment
        env_config = config.EnvConfig(args.debug)
        if args.human_num is not None:
            env_config.sim.human_num = args.human_num
        env = gym.make('CrowdSim-v0')
        e = 2
        se = 0

        env_config.env.dx_range = ec.exp.dx[e][se]
        env_config.env.dy_range = ec.exp.dy[e][se]
        env_config.sim.test_scenario = ec.exp.scenarios[e][se]
        env.configure(env_config)

        # if args.square:
        #     env.test_scenario = 'square_crossing'
        # if args.circle:
        #     env.test_scenario = 'circle_crossing'
        # if args.test_scenario is not None:
        #     env.test_scenario = args.test_scenario

        #env.test_scenario = args.scenario

        robot = Robot(env_config, 'robot')
        env.set_robot(robot)
        robot.time_step = env.time_step
        robot.set_policy(policy)
        explorer = Explorer(env, robot, device, None, gamma=0.9)

        train_config = config.TrainConfig(args.debug)
        epsilon_end = train_config.train.epsilon_end
        if not isinstance(robot.policy, ORCA):
            robot.policy.set_epsilon(epsilon_end)

        policy.set_phase(args.phase)
        policy.set_device(device)
        # set safety space for ORCA in non-cooperative simulation
        if isinstance(robot.policy, ORCA):
            if robot.visible:
                robot.policy.safety_space = args.safety_space
            else:
                robot.policy.safety_space = args.safety_space
            logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

        policy.set_env(env)
        robot.print_info()

        rewards = []
        #ob = env.reset(args.phase, args.test_case)
        time_start = time.time()
        ob = env.reset(args.phase, scenarios[e][se][0])
        print(scenarios[3][4][0])
        done = False
        last_pos = np.array(robot.get_position())
        while not done:
            rstate = robot.get_full_state()
            #logging.info("TEST ROBOT: px %.2f py %.2f vx %.2f vy %.2f v %.2f radius %.2f", rstate.px, rstate.py, rstate.vx, rstate.vy, np.sqrt(rstate.vx**2 + rstate.vy**2), rstate.radius)
            for human in env.humans:
                hstate = human.get_full_state()
                #logging.info("TEST HUMAN: px %.2f py %.2f vx %.2f vy %.2f v %.2f radius %.2f", hstate.px, hstate.py, hstate.vx, hstate.vy, np.sqrt(hstate.vx**2 + hstate.vy**2), hstate.radius)

            action = robot.act(ob, border=env.orca_border, baseline=args.baseline)
            #logging.info("ROBOT ACTION: vx %.2f vy %.2f", action.vx, action.vy)
            rstate2 = robot.get_full_state()
            #logging.info("POST ACT ROBOT: px %.2f py %.2f vx %.2f vy %.2f v %.2f radius %.2f", rstate2.px, rstate2.py, rstate2.vx, rstate2.vy, np.sqrt(rstate2.vx**2 + rstate2.vy**2), rstate2.radius)
            ob, _, done, info = env.step(action)
            #logging.info("STEP DONE", done)
            rewards.append(_)
            current_pos = np.array(robot.get_position())
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos
        gamma = 0.9
        cumulative_reward = sum([pow(gamma, t * robot.time_step * robot.v_pref)
            * reward for t, reward in enumerate(rewards)])
        
        time_end = time.time()
        print("TIME: ", time_end - time_start)

        if args.traj:
            env.render('traj', args.video_file)
        else:
            if args.video_dir is not None:
                if policy_config.name == 'gcn':
                    args.video_file = os.path.join(args.video_dir, policy_config.name + '_' + policy_config.gcn.similarity_function)
                else:
                    args.video_file = os.path.join(args.video_dir, args.video_file)
                #args.video_file = args.video_file + '_' + args.phase + '_' + str(args.test_case) + '.mp4'
            env.render('video', args.video_file)
        logging.info('It takes %.2f seconds to finish. Final status is %s, cumulative_reward is %f', env.global_time, info, cumulative_reward)
        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
    else:
        exp_stats_list = []
        if ec.exp.parameter_sweep is not None:
            for p1 in range(3):
                for p2 in range(3):
                    for p3 in range(3):
                        #if p3 > 2:
                        #        continue
                        for e in range(len(ec.exp.dx)):
                            if e != 0:
                                continue
                            for se in range(len(ec.exp.dx[e])):
                                if se != 1:
                                    continue
                                #print("EXPERIMENT: ", e, "SUBEXPERIMENT: ", se)
                                # configure environment
                                env_config = config.EnvConfig(args.debug)
                                env_config.env.dx_range = ec.exp.dx[e][se]
                                env_config.env.dy_range = ec.exp.dy[e][se]
                                env_config.env.randomize_attributes = ec.exp.randomize_attributes[e][se]
                                env_config.humans.num_sf = ec.exp.num_sf[e][se]
                                env_config.humans.num_orca = ec.exp.num_orca[e][se]
                                env_config.humans.num_static = ec.exp.num_static[e][se]
                                env_config.humans.num_linear = ec.exp.num_linear[e][se]
                                env_config.sim.test_scenario = ec.exp.scenarios[e][se]

                                if args.human_num is not None:
                                    env_config.sim.human_num = args.human_num
                                env = gym.make('CrowdSim-v0')
                                env.configure(env_config)

                                # if args.square:
                                #     env.test_scenario = 'square_crossing'
                                # if args.circle:
                                #     env.test_scenario = 'circle_crossing'
                                # if args.test_scenario is not None:
                                #     env.test_scenario = args.test_scenario

                                #env.test_scenario = args.scenario

                                robot = Robot(env_config, 'robot')
                                env.set_robot(robot)
                                robot.time_step = env.time_step
                                if ec.exp.parameter_sweep == 'sigma':
                                    param_list = ec.exp.sigma
                                    policy.model_predictor.sigma_h = param_list[p1]
                                    policy.model_predictor.sigma_s = param_list[p2]
                                    policy.model_predictor.sigma_r = param_list[p3]
                                elif ec.exp.parameter_sweep == 'q':
                                    param_list = ec.exp.q
                                    policy.model_predictor.q_obs = param_list[p1]
                                    policy.model_predictor.q_goal = param_list[p2]
                                    policy.model_predictor.q_wind = param_list[p3]
                                elif ec.exp.parameter_sweep == 'mppi':
                                    policy.set_mppi_params(ec.exp.noise[p1], ec.exp.samples[p2], ec.exp.horizon[p3])
                                robot.set_policy(policy)
                                explorer = Explorer(env, robot, device, None, gamma=0.9)

                                train_config = config.TrainConfig(args.debug)
                                epsilon_end = train_config.train.epsilon_end
                                if not isinstance(robot.policy, ORCA):
                                    robot.policy.set_epsilon(epsilon_end)

                                policy.set_phase(args.phase)
                                policy.set_device(device)
                                # set safety space for ORCA in non-cooperative simulation
                                if isinstance(robot.policy, ORCA):
                                    if robot.visible:
                                        robot.policy.safety_space = args.safety_space
                                    else:
                                        robot.policy.safety_space = args.safety_space
                                    #logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

                                policy.set_env(env)
                                #robot.print_info()
                                time_start = time.time()
                                stats, exp_stats = explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True, baseline=args.baseline)
                                time_end = time.time()
                                print("TIME TOTAL: ", time_end - time_start, " AVERAGE: ", (time_end - time_start) / env.case_size[args.phase])
                                exp_stats_list.append(exp_stats)
                                if args.plot_test_scenarios_hist:
                                    test_angle_seeds = np.array(env.test_scene_seeds)
                                    b = [i * 0.01 for i in range(101)]
                                    n, bins, patches = plt.hist(test_angle_seeds, b, facecolor='g')
                                    plt.savefig(os.path.join(args.model_dir, 'test_scene_hist.png'))
                                    plt.close()

                        #print("STATS LIST LEN:", len(exp_stats_list))
                        print("PARAMS: ", ec.exp.noise[p1], ec.exp.samples[p2], ec.exp.horizon[p3], " STATS: ", exp_stats)
        else:
            env_config = config.EnvConfig(args.debug)
            #scenarios = random_sequence(ec, env_config)
            for e in range(len(ec.exp.dx)):
                if e != 1:
                    continue
                for se in range(len(ec.exp.dx[e])):
                    if se != 0:
                        continue
                    print("EXPERIMENT: ", e, "SUBEXPERIMENT: ", se)
                    # configure environment
                    env_config.env.dx_range = ec.exp.dx[e][se]
                    env_config.env.dy_range = ec.exp.dy[e][se]
                    env_config.env.randomize_attributes = ec.exp.randomize_attributes[e][se]
                    env_config.humans.num_sf = ec.exp.num_sf[e][se]
                    env_config.humans.num_orca = ec.exp.num_orca[e][se]
                    env_config.humans.num_static = ec.exp.num_static[e][se]
                    env_config.humans.num_linear = ec.exp.num_linear[e][se]
                    env_config.sim.test_scenario = ec.exp.scenarios[e][se]

                    if args.human_num is not None:
                        env_config.sim.human_num = args.human_num
                    env = gym.make('CrowdSim-v0')
                    env.configure(env_config)

                    # if args.square:
                    #     env.test_scenario = 'square_crossing'
                    # if args.circle:
                    #     env.test_scenario = 'circle_crossing'
                    # if args.test_scenario is not None:
                    #     env.test_scenario = args.test_scenario

                    #env.test_scenario = args.scenario

                    robot = Robot(env_config, 'robot')
                    env.set_robot(robot)
                    robot.time_step = env.time_step
                    robot.set_policy(policy)
                    if scenarios is not None:
                        explorer = Explorer(env, robot, device, None, gamma=0.9, scenarios=scenarios[e][se])
                    else:
                        explorer = Explorer(env, robot, device, None, gamma=0.9)

                    train_config = config.TrainConfig(args.debug)
                    epsilon_end = train_config.train.epsilon_end
                    if not isinstance(robot.policy, ORCA):
                        robot.policy.set_epsilon(epsilon_end)

                    policy.set_phase(args.phase)
                    policy.set_device(device)
                    # set safety space for ORCA in non-cooperative simulation
                    if isinstance(robot.policy, ORCA):
                        if robot.visible:
                            robot.policy.safety_space = args.safety_space
                        else:
                            robot.policy.safety_space = args.safety_space
                        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

                    policy.set_env(env)
                    #robot.print_info()

                    stats, exp_stats = explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True, baseline=args.baseline)
                    exp_stats_list.append(exp_stats)
                    if args.plot_test_scenarios_hist:
                        test_angle_seeds = np.array(env.test_scene_seeds)
                        b = [i * 0.01 for i in range(101)]
                        n, bins, patches = plt.hist(test_angle_seeds, b, facecolor='g')
                        plt.savefig(os.path.join(args.model_dir, 'test_scene_hist.png'))
                        plt.close()

                    print("EXP STATS: ", exp_stats)
                print(exp_stats_list)

            #print("STATS LIST LEN:", len(exp_stats_list))
            print(exp_stats_list)
            #print("PARAMS: ", p1, p2, p3, " STATS: ", exp_stats)

def main(args):
    # configure logging and device
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    if args.model_dir is not None:
        if args.config is not None:
            config_file = args.config
        else:
            config_file = os.path.join(args.model_dir, 'config.py')
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
            logging.info('Loaded IL weights')
        elif args.rl:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                print(os.listdir(args.model_dir))
                model_weights = os.path.join(args.model_dir, sorted(os.listdir(args.model_dir))[-1])
            logging.info('Loaded RL weights')
        else:
            model_weights = os.path.join(args.model_dir, 'best_val.pth')
            logging.info('Loaded RL weights with best VAL')

    else:
        config_file = args.config

    spec = importlib.util.spec_from_file_location('config', config_file)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # configure policy
    policy_config = config.PolicyConfig(args.debug)
    policy = policy_factory[policy_config.name]()
    if args.planning_depth is not None:
        print("PLANNING DEPTH: ", args.planning_depth)
        policy_config.model_predictive_rl.do_action_clip = True
        policy_config.model_predictive_rl.planning_depth = args.planning_depth
    if args.planning_width is not None:
        print("PLANNING WIDTH: ", args.planning_width)
        policy_config.model_predictive_rl.do_action_clip = True
        policy_config.model_predictive_rl.planning_width = args.planning_width
    if args.sparse_search:
        policy_config.model_predictive_rl.sparse_search = True

    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.load_model(model_weights)

    # configure environment
    env_config = config.EnvConfig(args.debug)

    if args.human_num is not None:
        env_config.sim.human_num = args.human_num
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)

    # if args.square:
    #     env.test_scenario = 'square_crossing'
    # if args.circle:
    #     env.test_scenario = 'circle_crossing'
    # if args.test_scenario is not None:
    #     env.test_scenario = args.test_scenario

    env.test_scenario = args.scenario

    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    robot.time_step = env.time_step
    robot.set_policy(policy)
    explorer = Explorer(env, robot, device, None, gamma=0.9)

    train_config = config.TrainConfig(args.debug)
    epsilon_end = train_config.train.epsilon_end
    if not isinstance(robot.policy, ORCA):
        robot.policy.set_epsilon(epsilon_end)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = args.safety_space
        else:
            robot.policy.safety_space = args.safety_space
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()

    if args.visualize:
        rewards = []
        #ob = env.reset(args.phase, args.test_case)
        ob = env.reset(args.phase)
        done = False
        last_pos = np.array(robot.get_position())
        while not done:
            action = robot.act(ob, border=env.orca_border, baseline=args.baseline)
            ob, _, done, info = env.step(action)
            rewards.append(_)
            current_pos = np.array(robot.get_position())
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos
        gamma = 0.9
        cumulative_reward = sum([pow(gamma, t * robot.time_step * robot.v_pref)
             * reward for t, reward in enumerate(rewards)])

        if args.traj:
            env.render('traj', args.video_file)
        else:
            if args.video_dir is not None:
                if policy_config.name == 'gcn':
                    args.video_file = os.path.join(args.video_dir, policy_config.name + '_' + policy_config.gcn.similarity_function)
                else:
                    args.video_file = os.path.join(args.video_dir, policy_config.name)
                args.video_file = args.video_file + '_' + args.phase + '_' + str(args.test_case) + '.mp4'
            env.render('video', args.video_file)
        logging.info('It takes %.2f seconds to finish. Final status is %s, cumulative_reward is %f', env.global_time, info, cumulative_reward)
        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
    else:
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True, baseline=args.baseline)
        if args.plot_test_scenarios_hist:
            test_angle_seeds = np.array(env.test_scene_seeds)
            b = [i * 0.01 for i in range(101)]
            n, bins, patches = plt.hist(test_angle_seeds, b, facecolor='g')
            plt.savefig(os.path.join(args.model_dir, 'test_scene_hist.png'))
            plt.close()

def set_params_crossing_fixed(radius, agents, x_width, y_width, discomfort_dist):
    if np.random.random() > 0.5:
            sign = -1
    else:
        sign = 1
    while True:
        #px = np.random.random() * self.square_width * 0.5 * sign
        #py = (np.random.random() - 0.5) * self.square_width
        px = np.random.random() * x_width * 0.5 * sign
        py = (np.random.random() - 0.5) * y_width
        collide = False
        for agent in agents:
            #print(agent)
            if norm((px - agent[0], py - agent[1])) < radius + 0.3 + discomfort_dist:
                collide = True
                break
        if not collide:
            break
    while True:
        gx = np.random.random() * x_width * 0.5 * - sign
        gy = (np.random.random() - 0.5) * y_width
        collide = False
        for agent in agents:
            if norm((gx - agent[2], gy - agent[3])) < radius + 0.3 + discomfort_dist:
                collide = True
                break
        if not collide:
            break
    return px, py, gx, gy, 0, 0, 0

def set_params_passing_fixed(radius, agents, x_width, y_width, discomfort_dist):
    if np.random.random() > 0.5:
            sign = -1
    else:
        sign = 1
    while True:
        #px = np.random.random() * self.square_width * 0.5 * sign
        #py = (np.random.random() - 0.5) * self.square_width
        py = np.random.random() * y_width * 0.5 * sign
        px = (np.random.random() - 0.5) * x_width
        collide = False
        for agent in agents:
            if norm((px - agent[0], py - agent[1])) < radius + 0.3 + discomfort_dist:
                collide = True
                break
        if not collide:
            break
    while True:
        gy = np.random.random() * y_width * 0.5 * - sign
        if px > 0:
            gx = (np.random.random() * 0.5) * x_width
        else:
            gx = -1 * (np.random.random() * 0.5) * x_width
        #gx = (np.random.random() - 0.5) * self.x_width
        collide = False
        for agent in agents:
            if norm((gx - agent[2], gy - agent[3])) < radius + 0.3 + discomfort_dist:
                collide = True
                break
        if not collide:
            break
    return px, py, gx, gy, 0, 0, 0

def generate_human_state(agents, x_width, y_width, discomfort_dist, policy=None, current_scenario='passing_crossing'):
        #print("CURRENT SCENARIO: ", self.current_scenario)
        params = None
        if current_scenario == 'circle_crossing':
            while True:
                angle = np.random.random() * np.pi/2
                # add some noise to simulate all the possible cases robot could meet with human
                px_noise = (np.random.random() - 0.5) * 1.0
                py_noise = (np.random.random() - 0.5) * 1.0
                px = 4.0 * np.cos(angle) + px_noise
                py = 4.0 * np.sin(angle) + py_noise
                if np.random.random() > 0.5:
                    px = px * -1

                if np.random.random() > 0.5:
                    py = py * -1

                collide = False
                for agent in agents:
                    min_dist = 0.3 + 0.3 + discomfort_dist
                    if norm((px - agent[0], py - agent[1])) < min_dist or \
                            norm((px - agent[2], py - agent[3])) < min_dist:
                        collide = True
                        break
                if not collide:
                    break
            params = px, py, -px, -py, 0, 0, 0

        elif current_scenario == 'passing':
            params = set_params_passing_fixed(0.3, agents, x_width, y_width, discomfort_dist)

        elif current_scenario == 'crossing':
            params = set_params_crossing_fixed(0.3, agents, x_width, y_width, discomfort_dist)

        elif current_scenario == 'passing_crossing':
            if np.random.random() > 0.5:
                sign = -1
            else:
                sign = 1

            if sign == 1:
                params = set_params_passing_fixed(0.3, agents, x_width, y_width, discomfort_dist)
            else:
                params = set_params_crossing_fixed(0.3, agents, x_width, y_width, discomfort_dist)
            #print("PARAMS: ", params)

        elif current_scenario == 'random':
            #print("IN RANDOM")
            while True:
                py = (np.random.random() - 0.5) * y_width
                px = (np.random.random() - 0.5) * x_width
                collide = False
                for agent in agents:
                    if norm((px - agent[0], py - agent[1])) < 0.3 + 0.3 + discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            while True:
                gy = (np.random.random() - 0.5) * y_width
                gx = (np.random.random() - 0.5) * x_width
                collide = False
                for agent in agents:
                    if norm((gx - agent[2], gy - agent[3])) < 0.3 + 0.3 + discomfort_dist:
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

def generate_scenarios_fixed(length, radius, x_width, y_width, discomfort_dist, num_orca, num_sf, num_linear, num_static, current_scenario):
        states = []
        num_policies = {
            'orca' : num_orca,
            'socialforce' : num_sf,
            'linear' : num_linear,
            'static' : num_static
        }
        for i in range(length):
            states_i = {}
            agents = [(0, -4, 0, 4, 0, 0, np.pi / 2)]
            for policy in num_policies:
                for _ in range(num_policies[policy]):
                    if policy in states_i:
                        params = generate_human_state(agents, x_width, y_width, discomfort_dist, policy=policy, current_scenario=current_scenario)
                        states_i[policy].append(params)
                        agents.append(params)
                    else:
                        states_i[policy] = [generate_human_state(agents, x_width, y_width, discomfort_dist, policy=policy, current_scenario=current_scenario)]
            states.append(states_i)

        return states

def random_sequence(ec, seed=1000, length=500, radius=0.3, discomfort_dist=0.2):
    np.random.seed(seed)
    random.seed(seed)

    for i in range(100):
       print(np.random.random())

    scenarios = []

    for e in range(len(ec.exp.dx)):
        #if e != 0:
        #    continue
        scenarios_e = []
        for se in range(len(ec.exp.dx[e])):
            #if se != 2:
            #    continue
            print("EXPERIMENT: ", e, "SUBEXPERIMENT: ", se)
            # configure environment
            x_width = (ec.exp.dx[e][se][1] - ec.exp.dx[e][se][0]) - (2 * radius + 1e-2)
            y_width = (ec.exp.dy[e][se][1] - ec.exp.dy[e][se][0]) - (2 * radius + 1e-2)
            print("X Y WIDTH: ", x_width, y_width)
            num_sf = ec.exp.num_sf[e][se][0]
            num_orca = ec.exp.num_orca[e][se][0]
            num_static = ec.exp.num_static[e][se][0]
            num_linear = ec.exp.num_linear[e][se][0]
            test_scenario = ec.exp.scenarios[e][se]

            print(x_width, y_width, num_sf, num_orca, num_static, num_linear, test_scenario)

            scenarios_e.append(generate_scenarios_fixed(length, radius, x_width, y_width, discomfort_dist, num_orca, num_sf, num_linear, num_static, test_scenario))
        
        scenarios.append(scenarios_e)

    return scenarios


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--policy', type=str, default='model_predictive_rl')
    parser.add_argument('-m', '--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--rl', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('-v', '--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('-c', '--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--scenario', type=str, default='circle')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--video_dir', type=str, default='/home/socnav/arstr')
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--human_num', type=int, default=None)
    parser.add_argument('--safety_space', type=float, default=0.2)
    parser.add_argument('--test_scenario', type=str, default=None)
    parser.add_argument('--plot_test_scenarios_hist', default=True, action='store_true')
    parser.add_argument('-d', '--planning_depth', type=int, default=None)
    parser.add_argument('-w', '--planning_width', type=int, default=None)
    parser.add_argument('--sparse_search', default=False, action='store_true')
    parser.add_argument('--baseline', type=str, default=None)

    sys_args = parser.parse_args()

    main_experiments(sys_args)
