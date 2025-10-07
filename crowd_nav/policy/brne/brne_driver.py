import numpy as np
import rvo2
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY, ActionRot
# from . import brne as brne
from . import brne as brne

import logging
import math
from .traj_tracker import TrajTracker
import copy
import time
from crowd_nav.utils.data_collect import DATA



class BRNE_Driver(Policy):

    def __init__(self):
        super().__init__()
        self.trainable = False
        self.name = 'brne'
        self.kinematics = 'unicycle'
        self.multiagent_training = False
        self.sim = None
        self.plan_steps = 4
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
        self.cost_a1 = 7.0
        self.cost_a2 = 7.0
        self.cost_a3 = 5.0
        self.ped_sample_scale = 0.1
        self.corridor_y_min = -12.0
        self.corridor_y_max = 12.0
        self.cmd_tracker = TrajTracker(dt=self.dt, max_lin_vel=self.max_lin_vel, max_ang_vel=self.max_ang_vel)   # the class that converts waypoints to control commands
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


    def brne_cb(self, state):
        self_state = state.robot_state
        self.robot_goal = np.array([self_state.gx, self_state.gy])

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

        # print(f"CMD COUNTER: {self.cmd_counter}")

        # self.cmd_counter restricts the BRNE replanning to self.plan_steps
        # if there are other agents in the scene 
        if self.cmd_counter == 0 and num_agents > 1:
            ped_indices = np.argsort(dists2peds)[:num_agents-1]  # we only pick the N closest pedestrian to interact with
            robot_state = np.array([self_state.px,self_state.py,self_state.theta])

            x_pts = brne.mvn_sample_normal((num_agents-1) * self.num_samples, self.plan_steps, self.cov_Lmat)
            y_pts = brne.mvn_sample_normal((num_agents-1) * self.num_samples, self.plan_steps, self.cov_Lmat)

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

            goal = np.array([self_state.gx, self_state.gy])

            angle = st[2]
            if angle > 0.0:
                theta_a = angle - np.pi/2
            else:
                theta_a = angle + np.pi/2
            axis_vec = np.array([
                np.cos(theta_a), np.sin(theta_a)
            ])  

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

            tiles = np.tile(robot_state, reps=(self.num_samples,1)).T

            traj_essemble = brne.traj_sim_essemble( 
                tiles,
                ulist_essemble,
                self.dt
            )

            xtraj_samples[0:self.num_samples] = traj_essemble[:,0,:].T
            ytraj_samples[0:self.num_samples] = traj_essemble[:,1,:].T


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
            
            print(f"weight[0]: {weights[0]}")
            opt_cmds_1 = np.mean(ulist_essemble[:,:,0] * weights[0], axis=1)
            opt_cmds_2 = np.mean(ulist_essemble[:,:,1] * weights[0], axis=1)
            # logging.info("cmds: {}".format(np.mean(opt_cmds_1)))

            self.cmds = np.array([opt_cmds_1, opt_cmds_2]).T
            self.cmds_traj = self.cmd_tracker.sim_traj(robot_state, self.cmds)

            ped_trajs_x = np.zeros((num_agents-1, self.plan_steps))
            ped_trajs_y = np.zeros((num_agents-1, self.plan_steps))
            for i in range(num_agents-1):
                ped_trajs_x[i] = \
                    np.mean(xtraj_samples[(i+1)*self.num_samples : (i+2)*self.num_samples] * weights[i+1][:,np.newaxis], axis=0)
                ped_trajs_y[i] = \
                    np.mean(ytraj_samples[(i+1)*self.num_samples : (i+2)*self.num_samples] * weights[i+1][:,np.newaxis], axis=0)
            

        # if the ego agent is the only in the scene
        elif num_agents == 1:
            print("no peds")
            self.close_stop_flag = False
            robot_state = self_state
            st = np.array([robot_state.position[0], robot_state.position[1], robot_state.theta])
            print(f"ROBOT THETA: {st[2]}")
            goal = np.array([self_state.gx,self_state.gy])
            theta_a = st[2]
            if st[2] > 0.0:
                theta_a = st[2] - np.pi/2
            elif st[2] < 0.0:
                theta_a = st[2] + np.pi/2

            axis_vec = np.array([
                np.cos(theta_a), np.sin(theta_a)
            ])
            # rot = [[np.cos(np.pi/2), -np.sin(np.pi/2)],
            #        [np.sin(np.pi/2), np.cos(np.pi/2)]]
                   
            # axis_vec = rot @ axis_vec

            # TODO: There is a difference between what the robot thinks is 0 heading and what BRNE thinks is 0 heading
            # I think that BRNE thinks right is 0 and crowdnav thinks up is 0.
            ##
            vec2goal = np.array([goal[0] - st[0], goal[1] - st[1]])

            dist2goal = np.linalg.norm(vec2goal)
            proj_len = (axis_vec @ vec2goal) / (vec2goal @ vec2goal) * dist2goal
            # print(f"proj_len: {proj_len}")
            # print(f"dist2goal: {dist2goal}")
            if (proj_len == 0):
                proj_len = 10^(-10)
            radius = 0.5 * dist2goal / proj_len

            if st[2] > 0.0:
                ut = np.array([self.nominal_vel, -self.nominal_vel/radius])
            elif st[2] < 0.0:
                ut = np.array([self.nominal_vel, self.nominal_vel/radius])
            else:
                ut = np.array([self.nominal_vel, 0.0])
                
            nominal_cmds = np.tile(ut, reps=(self.plan_steps,1))

            nominal_vel = self.open_space_velocity

            ulist_essemble = brne.get_ulist_essemble(
                nominal_cmds, nominal_vel, self.max_ang_vel, self.num_samples
            )
            traj_essemble = brne.traj_sim_essemble(
                np.tile(st, reps=(self.num_samples,1)).T,
                ulist_essemble,
                self.dt
            )

            end_pose_essemble = traj_essemble[-1, 0:2, :].T
            dists2goal_essemble = np.linalg.norm(end_pose_essemble - goal, axis=1)
            opt_cmds = ulist_essemble[:, np.argmin(dists2goal_essemble), :]

            self.cmds = opt_cmds
            self.cmds_traj = self.cmd_tracker.sim_traj(st, self.cmds)

            if self.cmd_counter >= self.plan_steps:
                self.cmd_counter = 0

            self.define_trajectory(self.cmds_traj[:,0], self.cmds_traj[:,1])
            
        x_cmd = float(self.cmds[self.cmd_counter][0])
        rot_cmd = float(self.cmds[self.cmd_counter][1])
        print(f"x, rot: {x_cmd, rot_cmd}")

        self.cmd_counter += 1
        if self.cmd_counter >= self.plan_steps:
            self.cmd_counter = 0
        return ActionRot(x_cmd, rot_cmd)

        # print(self.cmds_traj)

    def predict(self, state, border=None, radius=None, baseline=None):
        start_time = time.time()
        cmd = self.brne_cb(state)
        tim = time.time() - start_time
        self.data_tracker.update_data(state,tim)
        # print(f"Single iteration runtime: {time.time() - start_time}")
        return cmd


    def define_trajectory(self, xs, ys):
        p = []

        for x,y in zip(xs, ys):
            pose = [x, y]
            p.append(pose)

        # logging.info("p: {}".format(p))
        return p