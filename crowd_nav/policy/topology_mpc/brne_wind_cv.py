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

# import dpilqr
# import scenarios

π = np.pi
g = 9.80665


class BRNEWINDCV(Policy):

    def __init__(self,config):
        super().__init__()
        self.name = 'brne_wind'
        self.trainable = False
        self.U = None
        self.MPC = vecMPC(config)
        self.kinematics = 'holonomic'
        self.multiagent_training = False
        self.sim = None
        self.plan_steps = 5
        self.num_samples = 4
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




    def configure(self, config):
        assert True


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
            # # TODO: Check theta frame/range with print statements
            # axis_vec = np.array([
            #     np.cos(angle), np.sin(angle)
            # ])
            vec2goal = np.array([goal[0] - st[0], goal[1] - st[1]])
            # logging.info("vec2goal: {}".format(vec2goal))
            dist2goal = np.linalg.norm(vec2goal)
            proj_len = (axis_vec @ vec2goal) / (vec2goal @ vec2goal) * dist2goal
            # proj_len = (np.dot(axis_vec, vec2goal)) / (np.dot(vec2goal, vec2goal)) * dist2goal
            # logging.info("proj_len: {}".format(proj_len))
            # if (proj_len == 0):
            #     proj_len = 0.001
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

            # generate sample weight mask for the closest pedestrian
            robot_xtrajs = traj_essemble[:,0,:].T
            robot_ytrajs = traj_essemble[:,1,:].T
            robot_samples2ped = (robot_xtrajs - closest_ped_pos[0])**2 + (robot_ytrajs - closest_ped_pos[1])**2
            robot_samples2ped = np.min(np.sqrt(robot_samples2ped), axis=1)
            safety_mask = (robot_samples2ped > self.close_stop_threshold).astype(float)
            safety_samples_percent = safety_mask.mean() * 100
            # self.get_logger().debug('dist 2 ped: {:.2f} m'.format(closest_dist2ped))
            
            self.close_stop_flag = False
            if np.max(safety_mask) == 0.0:
                safety_mask = np.ones_like(safety_mask)
                self.close_stop_flag = True
            # self.get_logger().debug('safety mask: {}'.format(safety_mask))

            # BRNE OPTIMIZATION HERE !!!
            weights = brne.brne_nav(
                xtraj_samples, ytraj_samples,
                num_agents, self.plan_steps, self.num_samples,
                self.cost_a1, self.cost_a2, self.cost_a3, self.ped_sample_scale,
                self.corridor_y_min, self.corridor_y_max
            ) 

            # print(f"weights: {weights}")
            if (np.mean(weights[0]) != 0):
                weights[0] /= np.mean(weights[0])
            
            opt_cmds_1 = np.mean(ulist_essemble[:,:,0] * weights[0], axis=1)
            opt_cmds_2 = np.mean(ulist_essemble[:,:,1] * weights[0], axis=1)

            self.cmds = np.array([opt_cmds_1, opt_cmds_2]).T
            # print(f"self.cmds: {self.cmds}")
            self.cmds_traj = self.cmd_tracker.sim_traj(robot_state, self.cmds)


            ped_trajs_x = np.zeros((num_agents-1, self.plan_steps))
            ped_trajs_y = np.zeros((num_agents-1, self.plan_steps))
            for i in range(num_agents-1):
                ped_trajs_x[i] = \
                    np.mean(xtraj_samples[(i+1)*self.num_samples : (i+2)*self.num_samples] * weights[i+1][:,np.newaxis], axis=0)
                ped_trajs_y[i] = \
                    np.mean(ytraj_samples[(i+1)*self.num_samples : (i+2)*self.num_samples] * weights[i+1][:,np.newaxis], axis=0)
            traj = self.cmd_tracker.sim_traj(robot_state, self.cmds)
            return traj
        
        x_cmd = float(self.cmds[self.cmd_counter][0])
        rot_cmd = float(self.cmds[self.cmd_counter][1])

        self.cmd_counter += 1
        if self.cmd_counter >= self.plan_steps:
            self.cmd_counter = 0
        # self.get_logger().debug(f'current control: [{cmd.linear.x}, {cmd.angular.z}]')
        return None



    # rollout robot trajectory with BNRE, rollout constant velocity pedestrian trajectory 
    def brne_cv_rollout(self, state):

        rob_traj = self.brne_cb(state)
        X = np.zeros((len(rob_traj),len(state.human_states)+1, 2))
        s = state.robot_state
        angle = np.arctan2(s.gy-s.py, s.gx-s.px)
        v_norm = np.sqrt(s.vx**2 + s.vy**2)

        for k in range(len(rob_traj)):
            X[k][0][0] = rob_traj[k][0]
            X[k][0][1] = rob_traj[k][1]
        

        for k in range(len(rob_traj)):
            for i, s in enumerate(state.human_states):
                X[k][i+1][0] = s.px + s.vx * k * 0.25 # TODO: make this not harcoded
                X[k][i+1][1] = s.py + s.vy * k * 0.25

        return X
    

    # 
    def local_controller(self, state):
        X = self.brne_cv_rollout(state)
        w_vec = np.zeros(len(state.human_states))

        # print(len(X[0])/3)
        if X is not None:
            for i in range(1, len(X[0])):
                theta_end = np.arctan2(X[1][i][1] - X[1][0][1], X[1][i][0] - X[1][0][0])
                theta_start = np.arctan2(X[0][i][1] - X[0][0][1], X[0][i][0] - X[0][0][0])
                w = (theta_end - theta_start)/(2*np.pi)
                w_vec[i-1] = w

        print(f"BRNE (cv ped) wind_vec: {w_vec}")
        # predictions = self.MPC.get_predictions(state, actions) # N x S x T' x H x 4
        self.brne_cb(state)
        ctrl = self.MPC.predict(state, w_vec)
        # print(f"sza!: {ctrl}")
        sum = (abs(ctrl[0])+ abs(ctrl[0]))
        factor = 0.6
        if (sum > factor) :
            ux_norm = (ctrl[0]  / sum) * factor
            uy_norm = (ctrl[1]  / sum) * factor
        else:
            ux_norm = ctrl[0]
            uy_norm = ctrl[1]

        return ActionXY(ux_norm, uy_norm)

    # def predict(self,state):
    def predict(self, state, border=None, radius=None, baseline=None):
        start_time = time.time()
        control = self.local_controller(state)
        print(f"Single Iterationn Runtime: {time.time() - start_time}")
        return control