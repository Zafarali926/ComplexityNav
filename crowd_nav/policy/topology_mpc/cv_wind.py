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

import time

# import dpilqr
# import scenarios

π = np.pi
g = 9.80665


class CVWIND(Policy):

    def __init__(self,config):
        super().__init__()
        self.name = 'cv_wind'
        self.trainable = False
        self.kinematics = 'holonomic'
        self.multiagent_training = False
        self.sim = None
        self.U = None
        self.MPC = vecMPC(config)
  

    def cv_rollout(self, state):
        X = np.zeros((2,len(state.human_states)+1, 2))
        s = state.robot_state
        angle = np.arctan2(s.gy-s.py, s.gx-s.px)
        v_norm = np.sqrt(s.vx**2 + s.vy**2)

        X[0][0][0] = s.px
        X[0][0][1] = s.py
        X[1][0][0] = s.px + v_norm * np.cos(angle) * 7 * 0.25 
        X[1][0][1] = s.py + v_norm * np.sin(angle) * 7 * 0.25 

        for i, s in enumerate(state.human_states):
            X[0][i+1][0] = s.px
            X[0][i+1][1] = s.py
            X[1][i+1][0] = s.px + s.vx * 7 * 0.25 # TODO: make this not harcoded
            X[1][i+1][1] = s.py + s.vy * 7 * 0.25

        return X
    


    # uses constant velocity rollout to define topological constraints for MPC
    def local_controller(self, state):
        X = self.cv_rollout(state)
        w_vec = np.zeros(len(state.human_states))

        # print(len(X[0])/3)
        for i in range(1, len(X[0])):
            theta_end = np.arctan2(X[1][i][1] - X[1][0][1], X[1][i][0] - X[1][0][0])
            theta_start = np.arctan2(X[0][i][1] - X[0][0][1], X[0][i][0] - X[0][0][0])
            w = (theta_end - theta_start)/(2*np.pi)
            w_vec[i-1] = w

        print(f"CV wind_vec: {w_vec}")
        # predictions = self.MPC.get_predictions(state, actions) # N x S x T' x H x 4

        ctrl = self.MPC.predict(state, w_vec)
        print(f"sza!: {ctrl}")
        sum = (abs(ctrl[0])+ abs(ctrl[0]))
        factor = 0.6
        if (sum > factor) :
            ux_norm = (ctrl[0]  / sum) * factor
            uy_norm = (ctrl[1]  / sum) * factor
        else:
            ux_norm = ctrl[0]
            uy_norm = ctrl[1]

        return ActionXY(ux_norm, uy_norm)

    def predict(self, state, border=None, radius=None, baseline=None):
        start_time = time.time()
        control = self.local_controller(state)
        print(f"Single Iterationn Runtime: {time.time() - start_time}")
        return control