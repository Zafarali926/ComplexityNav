#!/usr/bin/env python

"""Collection of examples that demonstrate the functionality of
distributed-potential-ilqr in various scenarios.

Including:
 - single unicycle
 - single quadcopter (6D)
 - two quads with one human
 - random multi-agent simulation

"""

import rvo2
from crowd_sim.envs.utils.action import ActionXY, ActionRot

import numpy as np
import matplotlib.pyplot as plt
from crowd_sim.envs.policy.policy import Policy
from crowd_nav.policy.vecMPC.controller_wind import vecMPC

from crowd_nav.policy.brne import brne as brne

import logging
import math
from crowd_nav.policy.brne.traj_tracker import TrajTracker
import copy
import time
import itertools

from crowd_nav.utils.data_collect import DATA

# import dpilqr
# import scenarios

π = np.pi
g = 9.80665


class BRNEWIND(Policy):

    def __init__(self,config):
        super().__init__()
        self.name = 'brne_wind'
        self.trainable = False
        self.U = None
        self.MPC = vecMPC(config)
        self.kinematics = 'holonomic'
        self.multiagent_training = False
        self.sim = None
        self.plan_steps = 5 # must be <= 7 for MPC
        self.num_samples = 5
        self.a1 = 10
        self.a2 = 10
        self.a3 = 10
        self.brne_activate_threshold = 12
        self.dt = 0.25
        self.nominal_vel = 1.0
        self.max_lin_vel = 1.0
        self.max_ang_vel = 0.3
        self.close_stop_threshold = 0.5
        self.cost_a1 = 2.0
        self.cost_a2 = 7.0
        self.cost_a3 = 5.0
        self.ped_sample_scale = 0.1
        self.corridor_y_min = -12.0
        self.corridor_y_max = 12.0
        self.cmd_tracker = TrajTracker(dt=self.dt, max_lin_vel=self.max_lin_vel, max_ang_vel=self.max_ang_vel)   # the class that converts way points to control commands
        self.robot_goal = None
        self.cmd_counter = 0
        self.open_space_velocity = 0.6
        self.num_agents = 13 # maximum number of agents considered
        tlist = np.arange(self.plan_steps) * self.dt
        train_ts = np.array([tlist[0]])
        train_noise = np.array([1e-04])
        test_ts = tlist
        self.kernel_a1 = 0.2
        self.kernel_a2 = 0.2
        self.cov_Lmat, self.cov_mat = brne.get_Lmat_nb(train_ts, test_ts, train_noise, self.kernel_a1, self.kernel_a2)
        self.data_tracker = DATA()

    def configure(self, config):
        assert True


    # calculate the winding number w.r.t. one human
    def calc_w(self,rob_x, rob_y, hum_x, hum_y):
        w = 0

        for i in range(1, len(rob_x)):
            theta1 = np.arctan2(rob_y[i] - hum_y[i], rob_x[i] - hum_x[i])
            theta2 = np.arctan2(rob_y[i-1] - hum_y[i-1], rob_x[i-1] - hum_x[i-1])
            w += theta1-theta2

        return w


    # compute distribution over winding vectors based on BRNE rollout
    def wind_distribution(self, state, x_traj_samples, y_traj_samples, weights):
        wind_matrix = np.zeros((self.num_samples, len(state.human_states), self.num_samples)) # (robot sample) x (human index) x (human sample)
        weight_matrix = np.zeros((self.num_samples, len(state.human_states), self.num_samples))
        # print(f"weights size: {weights.shape}")


        for i in range(self.num_samples): # for each robot traj sample
            for j in range(len(state.human_states)): # for each human
                for k in range(self.num_samples): # for each human traj sample
                    w = self.calc_w(x_traj_samples[i], y_traj_samples[i], x_traj_samples[(j+1)*(self.num_samples) + k], y_traj_samples[(j+1)*(self.num_samples) + k])
                    if (w < 0):
                        wind_matrix[i,j,k] = -1
                    elif (w > 0):
                        wind_matrix[i,j,k] = 1

                    weight_matrix[i,j,k] = weights[0, i] + weights[1+j, k]
        
        combinations = list(itertools.product(range(0, self.num_samples), repeat=(len(state.human_states)+1)))
        wind_vecs = np.zeros((len(combinations), len(state.human_states)))
        all_weights = np.zeros(len(combinations))
        for i in range(len(combinations)):
            for j in range(1, len(state.human_states)+1):
                wind_vecs[i][j-1] = wind_matrix[combinations[i][0]][j-1][combinations[i][j]]
                all_weights[i] += weight_matrix[combinations[i][0]][j-1][combinations[i][j]] # note that this considers the robot sample weight twice


        wind_list = []
        wind_list.append(wind_vecs[0])

        weight_list = []
        weight_list.append(all_weights[0])

        is_in = False

        for i in range(1,(len(wind_vecs))):
            vec = wind_vecs[i]
            for j in range(len(wind_list)):
                if (vec == wind_list[j]).all():
                    weight_list[j] += all_weights[i]
                    is_in = True
                    break

            if not is_in:
                wind_list.append(vec)
                weight_list.append(all_weights[i])
            else:
                is_in = False
        
        weight_array = np.array(weight_list)
        weight_norm = np.linalg.norm(weight_array)
        weight_array = weight_array/weight_norm

        # print(f"wind_list: {wind_list}")
        print(f"weight_array: {weight_array}")
        return wind_list, weight_array
                


        

    def brne_cb(self, state):
        self_state = state.robot_state
        self.robot_goal = np.array([self_state.gx, self_state.gy])

        # BEGIN BRNE
        ped_info_list = []
        dists2peds = []

        # we go through each perceived pedestrian and save the information
        for ped in state.human_states:
            dist2ped = np.sqrt((self_state.px-ped.px)**2 + (self_state.py-ped.py)**2)
            if dist2ped < self.brne_activate_threshold:  # only consider pedestrians within the activate threshold
                ped_info = np.array([
                    ped.px, ped.py, ped.vx, ped.vy
                ])
                ped_info_list.append(ped_info)

                dists2peds.append(dist2ped)

        ped_info_list = np.array(ped_info_list)
        self.num_peds = len(ped_info_list)

        dists2peds = np.array(dists2peds)

        # compute how many pedestrians we are actually interacting with
        num_agents = np.minimum(self.num_peds+1, self.num_agents)

        print(f"CMD COUNTER: {self.cmd_counter}")



        if self.cmd_counter == 0 and num_agents > 1:
            # print("peds")

            ped_indices = np.argsort(dists2peds)[:num_agents-1]  # we only pick the N closest pedestrian to interact with
            robot_state = np.array([self_state.px,self_state.py,self_state.theta])

            x_pts = brne.mvn_sample_normal((num_agents-1) * self.num_samples, self.plan_steps, self.cov_Lmat)
            y_pts = brne.mvn_sample_normal((num_agents-1) * self.num_samples, self.plan_steps, self.cov_Lmat)

            # self.get_logger().info(f'X and y pts shape {x_pts.shape} {y_pts.shape}')

            # ctrl space configuration here
            xtraj_samples = np.zeros((
                num_agents * self.num_samples, self.plan_steps
            ))
            ytraj_samples = np.zeros((
                num_agents * self.num_samples, self.plan_steps
            ))

            closest_dist2ped = 100.0
            closest_ped_pos = np.zeros(2) + 100.0
            for i, ped_id in enumerate(ped_indices):
                ped_pos = ped_info_list[ped_id][:2]
                ped_vel = ped_info_list[ped_id][2:]
                speed_factor = np.linalg.norm(ped_vel)
                ped_xmean = ped_pos[0] + np.arange(self.plan_steps) * self.dt * ped_vel[0]
                ped_ymean = ped_pos[1] + np.arange(self.plan_steps) * self.dt * ped_vel[1]


                xtraj_samples[(i+1)*self.num_samples : (i+2)*self.num_samples] = \
                    x_pts[i*self.num_samples : (i+1)*self.num_samples] * speed_factor + ped_xmean
                ytraj_samples[(i+1)*self.num_samples : (i+2)*self.num_samples] = \
                    y_pts[i*self.num_samples : (i+1)*self.num_samples] * speed_factor + ped_ymean
                
                dist2ped = np.linalg.norm([
                    robot_state[:2] - ped_pos[:2] # questionable translation from ros
                ])
                if dist2ped < closest_dist2ped:
                    closest_dist2ped = dist2ped
                    closest_ped_pos = ped_pos.copy()


            st = copy.copy(robot_state)

            # if self.robot_goal is None:
            #     goal = np.array([6.0, 0.0])
            # else:
            #     goal = self.robot_goal[:2]

            goal = np.array([self_state.gx, self_state.gy])
            # logging.info("goal: {}".format(goal))

            angle = st[2]
            if angle > 0.0:
                theta_a = angle - np.pi/2
            else:
                theta_a = angle + np.pi/2
            axis_vec = np.array([
                np.cos(theta_a), np.sin(theta_a)
            ])  

            vec2goal = np.array([goal[0] - st[0], goal[1] - st[1]])
            dist2goal = np.linalg.norm(vec2goal)
            proj_len = (axis_vec @ vec2goal) / (vec2goal @ vec2goal) * dist2goal
            radius = 0.5 * dist2goal / proj_len 
            # print(proj_len)

            # print(f"radius: {radius}")
            if angle > 0.0:
                ut = np.array([self.nominal_vel, -self.nominal_vel/radius])
            else:
                ut = np.array([self.nominal_vel, self.nominal_vel/radius])
            nominal_cmds = np.tile(ut, reps=(self.plan_steps,1))
            # self.get_logger().info(f"Nominal commands {nominal_cmds.shape}\n{nominal_cmds}")
            ulist_essemble = brne.get_ulist_essemble(
                nominal_cmds, self.max_lin_vel, self.max_ang_vel, self.num_samples
            )
            # print("hello")
            # self.get_logger().info(f"ulist {ulist_essemble.shape}\n{ulist_essemble}")
            # print(f"essemble {ulist_essemble}")
            tiles = np.tile(robot_state, reps=(self.num_samples,1)).T

            traj_essemble = brne.traj_sim_essemble( 
                tiles,
                ulist_essemble,
                self.dt
            )

            # print(traj_essemble)
            # logging.info("essemble: {}".format(traj_essemble))

            xtraj_samples[0:self.num_samples] = traj_essemble[:,0,:].T
            ytraj_samples[0:self.num_samples] = traj_essemble[:,1,:].T
            # print(f"xtraj_samples: {xtraj_samples.shape}")

            # generate sample weight mask for the closest pedestrian
            robot_xtrajs = traj_essemble[:,0,:].T
            robot_ytrajs = traj_essemble[:,1,:].T
            robot_samples2ped = (robot_xtrajs - closest_ped_pos[0])**2 + (robot_ytrajs - closest_ped_pos[1])**2
            robot_samples2ped = np.min(np.sqrt(robot_samples2ped), axis=1)
            safety_mask = (robot_samples2ped > self.close_stop_threshold).astype(float)
            safety_samples_percent = safety_mask.mean() * 100
            # self.get_logger().debug('percent of safe samples: {:.2f}%'.format(safety_samples_percent))
            # self.get_logger().debug('dist 2 ped: {:.2f} m'.format(closest_dist2ped))

            
            self.close_stop_flag = False
            if np.max(safety_mask) == 0.0:
                safety_mask = np.ones_like(safety_mask)
                self.close_stop_flag = True

            # BRNE OPTIMIZATION HERE !!!
            weights = brne.brne_nav(
                xtraj_samples, ytraj_samples,
                num_agents, self.plan_steps, self.num_samples,
                self.cost_a1, self.cost_a2, self.cost_a3, self.ped_sample_scale,
                self.corridor_y_min, self.corridor_y_max
            ) 
            # print(f"u_list_essemble: {ulist_essemble}")

            # print(f"weights: {weights}")
            if (np.mean(weights[0]) != 0):
                weights[0] /= np.mean(weights[0])
            
            # print(f"weight[0]: {weights[0]}")
            # print(f"ulist_essemble[:,:,1]: {ulist_essemble[:,:,1]}")
            opt_cmds_1 = np.mean(ulist_essemble[:,:,0] * weights[0], axis=1)
            opt_cmds_2 = np.mean(ulist_essemble[:,:,1] * weights[0], axis=1)

            self.cmds = np.array([opt_cmds_1, opt_cmds_2]).T
            # print(f"self.cmds: {self.cmds}")
            # self.cmds_traj = self.cmd_tracker.sim_traj(robot_state, self.cmds)
            self.cmds_traj = self.cmd_tracker.sim_traj(robot_state, self.cmds)


            all_trajs_x = np.zeros((num_agents, self.plan_steps))
            all_trajs_y = np.zeros((num_agents, self.plan_steps))
            for i in range(num_agents):
                all_trajs_x[i] = \
                    np.mean(xtraj_samples[(i)*self.num_samples : (i+1)*self.num_samples] * weights[i][:,np.newaxis], axis=0)
                all_trajs_y[i] = \
                    np.mean(ytraj_samples[(i)*self.num_samples : (i+1)*self.num_samples] * weights[i][:,np.newaxis], axis=0)
            # print(f"traj:\n{all_trajs_x}")

            
            wind_list, weight_array = self.wind_distribution(state, xtraj_samples, ytraj_samples, weights)
            # print(f"weights: {weights}")
            return all_trajs_x, all_trajs_y, opt_cmds_1, opt_cmds_2, wind_list, weight_array
        
        x_cmd = float(self.cmds[self.cmd_counter][0])
        rot_cmd = float(self.cmds[self.cmd_counter][1])
        # print(f"x, rot: {x_cmd, rot_cmd}")

        self.cmd_counter += 1
        if self.cmd_counter >= self.plan_steps:
            self.cmd_counter = 0
        # self.get_logger().debug(f'current control: [{cmd.linear.x}, {cmd.angular.z}]')
        return None



    def brne_rollout(self, state, traj_set_x, traj_set_y):

        X = np.zeros((len(traj_set_x),self.plan_steps, 2))
        s = state.robot_state
        angle = np.arctan2(s.gy-s.py, s.gx-s.px)
        v_norm = np.sqrt(s.vx**2 + s.vy**2)

        for k in range(len(traj_set_x)):
            for j in range(len(traj_set_x[0])):
                X[k][j][0] = traj_set_x[k][j]
                X[k][j][1] = traj_set_y[k][j]
        
        # print(f"X: {X}")

        return X
    


    # uses BRNE rollout to define topological constraints for MPC
    def local_controller(self, state):

        all_trajs_x, all_trajs_y, opt_cmds_1, opt_cmds_2, wind_list, weight_array= self.brne_cb(state)
        X = self.brne_rollout(state, all_trajs_x, all_trajs_y)
        U = np.zeros((len(opt_cmds_1), 2))
        U[:,0] = opt_cmds_1
        U[:,1] = opt_cmds_2

        w_vec = np.zeros(len(X)-1)

        if X is not None:
            # for j in range(1, len(X[0])):
            for i in range(1, len(X)):
                for j in range(1, len(X[0])):
                    theta1 = np.arctan2(X[i][j][1] - X[0][j][1], X[i][j][0] - X[0][j][0])
                    theta2 = np.arctan2(X[i][j-1][1] - X[0][j-1][1], X[i][j-1][0] - X[0][j-1][0])

                    w_vec[i-1] += theta1-theta2
        
        w_vec = w_vec / (2*np.pi)


        # print(f"BRNE wind_vec: {w_vec}")
        # predictions = self.MPC.get_predictions(state, actions) # N x S x T' x H x 4
        
        self.brne_cb(state)
        predictions = X[1:].copy()
        ctrl = self.MPC.predict(state, wind_list, weight_array, predictions = predictions) # distribution predict

        sum = (abs(ctrl[0])+ abs(ctrl[0]))
        factor = 0.6
        if (sum > factor) :
            ux_norm = (ctrl[0]  / sum) * factor
            uy_norm = (ctrl[1]  / sum) * factor
        else:
            ux_norm = ctrl[0]
            uy_norm = ctrl[1]
        
        print(f"Ctrl: {ux_norm, uy_norm}")
        return ActionXY(ux_norm, uy_norm)


    # def predict(self,state):
    def predict(self, state, border=None, radius=None, baseline=None):
        start_time = time.time()
        control = self.local_controller(state)

        tim = time.time() - start_time
        self.data_tracker.update_data(state,tim)
        # print(f"Single Iterationn Runtime: {time.time() - start_time}")
        return control