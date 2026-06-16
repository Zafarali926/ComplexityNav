import numpy as np
import torch
import socialforce
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class SocialForce(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'socialforce'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.initial_speed = 0.5
        self.v0 = 10
        self.sigma = 0.3
        self.sim = None
        self.time_step = 0.25

    def configure(self, config):
        return

    def set_phase(self, phase):
        return

    def _socialforce_row(self, px, py, vx, vy, gx, gy, v_pref):
        """Build a 10-dim socialforce state: pos, vel, accel, dest, tau, preferred speed."""
        pref = float(v_pref) if v_pref is not None else self.initial_speed
        if pref < 1e-3:
            pref = 1e-3
        return [px, py, vx, vy, 0.0, 0.0, gx, gy, 0.5, pref]

    def _preferred_speed(self, agent_state, fallback):
        """FullState has v_pref; ObservableState (robot obs of humans) does not."""
        return getattr(agent_state, 'v_pref', fallback)

    def predict(self, state, border=None):
        """

        :param state:
        :return:
        """
        sf_state = []
        self_state = state.robot_state
        sf_state.append(self._socialforce_row(
            self_state.px, self_state.py, self_state.vx, self_state.vy,
            self_state.gx, self_state.gy, self_state.v_pref))
        for human_state in state.human_states:
            # approximate desired direction with current velocity
            if human_state.vx == 0 and human_state.vy == 0:
                gx = np.random.random()
                gy = np.random.random()
            else:
                gx = human_state.px + human_state.vx
                gy = human_state.py + human_state.vy
            sf_state.append(self._socialforce_row(
                human_state.px, human_state.py, human_state.vx, human_state.vy,
                gx, gy, self._preferred_speed(human_state, self_state.v_pref)))
        sim = socialforce.Simulator(delta_t=self.time_step)
        trajectory = sim.run(torch.tensor(sf_state, dtype=torch.float32), n_steps=1)
        new_state = trajectory[-1]
        action = ActionXY(new_state[0, 2].item(), new_state[0, 3].item())
        #if border is not None and self.outside_check([sim.state[0, 0], sim.state[0, 1]], self_state.radius, border):
        #    action = ActionXY(0.0, 0.0)

        self.last_state = state

        return action


class CentralizedSocialForce(SocialForce):
    """
    Centralized socialforce, a bit different from decentralized socialforce, where the goal position of other agents is
    set to be (0, 0)
    """
    def __init__(self):
        super().__init__()
        self.time_step = 0.25

    def outside_check(self, position, radius, obstacle):
        left = position[0] - radius < obstacle[0][0]
        right = position[0] + radius > obstacle[1][0]
        below = position[1] - radius < obstacle[2][1]
        above = position[1] + radius > obstacle[1][1]
        if ((left or right) or (above or below)):
            return True

        return False

    def predict(self, state, border=None):
        sf_state = []
        for agent_state in state:
            sf_state.append(self._socialforce_row(
                agent_state.px, agent_state.py, agent_state.vx, agent_state.vy,
                agent_state.gx, agent_state.gy, agent_state.v_pref))

        sim = socialforce.Simulator(delta_t=self.time_step)
        trajectory = sim.run(torch.tensor(sf_state, dtype=torch.float32), n_steps=1)
        new_state = trajectory[-1]
        actions = [
            ActionXY(new_state[i, 2].item(), new_state[i, 3].item())
            for i in range(len(state))]
        del sim
        return actions
