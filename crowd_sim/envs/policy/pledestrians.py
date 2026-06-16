import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY

_EPSILON = 1e-5
_MAX_COST = 1e12


def _time_to_collision(px, py, vx, vy, radius, opx, opy, ovx, ovy, oradius, safety_space=0.01):
    """Earliest positive collision time between two disc agents, or inf if none."""
    R = radius + oradius + 2.0 * safety_space
    x = np.array([px - opx, py - opy], dtype=np.float64)
    v = np.array([vx - ovx, vy - ovy], dtype=np.float64)
    a = np.dot(v, v)
    if a < _EPSILON:
        return np.inf

    b = -np.dot(x, v)
    c = np.dot(x, x) - R * R
    if c <= 0:
        return 0.0

    d = b * b - a * c
    if d < 0:
        return np.inf

    sqrt_d = np.sqrt(d)
    tau = (b - sqrt_d) / a
    if tau < 0:
        return np.inf
    return tau


def _time_to_first_collision(px, py, vx, vy, radius, neighbors, neighbor_dist, safety_space):
    min_ttc = np.inf
    for other in neighbors:
        dist = np.hypot(px - other.px, py - other.py)
        if dist > neighbor_dist:
            continue
        ttc = _time_to_collision(
            px, py, vx, vy, radius, other.px, other.py, other.vx, other.vy, other.radius,
            safety_space=safety_space)
        if ttc < min_ttc:
            min_ttc = ttc
    return min_ttc


def _get_cost(px, py, gx, gy, vx, vy, radius, neighbors, neighbor_dist, w_a, w_b, t_min, t_max, safety_space):
    """Cost from Guy et al. (2010), ported from UMANS PLEdestrians.cpp."""
    ttc = _time_to_first_collision(px, py, vx, vy, radius, neighbors, neighbor_dist, safety_space)
    if ttc < t_min:
        return _MAX_COST

    speed_sq = vx * vx + vy * vy
    gx_off = gx - px - t_max * vx
    gy_off = gy - py - t_max * vy
    goal_dist = np.hypot(gx_off, gy_off)
    return t_max * (w_a + w_b * speed_sq) + 2.0 * goal_dist * np.sqrt(w_a * w_b)


def _goal_velocity(px, py, gx, gy, v_pref):
    direction = np.array([gx - px, gy - py], dtype=np.float64)
    dist = np.linalg.norm(direction)
    if dist < _EPSILON:
        return 0.0, 0.0
    return direction[0] / dist * v_pref, direction[1] / dist * v_pref


def _sample_velocities(px, py, gx, gy, v_pref, n_directions=16, n_speeds=5):
    if n_speeds <= 1:
        speeds = [v_pref]
    else:
        speeds = [v_pref * (i + 1) / n_speeds for i in range(n_speeds)]
    samples = [(0.0, 0.0)]
    samples.append(_goal_velocity(px, py, gx, gy, v_pref))
    for speed in speeds:
        for i in range(n_directions):
            theta = 2.0 * np.pi * i / n_directions
            samples.append((speed * np.cos(theta), speed * np.sin(theta)))
    return samples


def _predict_full_state(agent_state, neighbors, w_a, w_b, t_min, t_max, neighbor_dist, n_directions, n_speeds,
                        safety_space):
    best_cost = _MAX_COST
    best_vx, best_vy = 0.0, 0.0
    for vx, vy in _sample_velocities(
            agent_state.px, agent_state.py, agent_state.gx, agent_state.gy,
            agent_state.v_pref, n_directions, n_speeds):
        cost = _get_cost(
            agent_state.px, agent_state.py, agent_state.gx, agent_state.gy,
            vx, vy, agent_state.radius, neighbors, neighbor_dist,
            w_a, w_b, t_min, t_max, safety_space)
        if cost < best_cost:
            best_cost = cost
            best_vx, best_vy = vx, vy
    if best_cost >= _MAX_COST:
        best_vx, best_vy = _goal_velocity(
            agent_state.px, agent_state.py, agent_state.gx, agent_state.gy, agent_state.v_pref)
    return ActionXY(best_vx, best_vy)


class PLEdestrians(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'pledestrians'
        self.trainable = False
        self.multiagent_training = True
        self.kinematics = 'holonomic'
        self.w_a = 2.23
        self.w_b = 1.26
        self.t_min = 0.5
        self.t_max = 3.0
        self.neighbor_dist = 10.0
        self.safety_space = 0.01  # matches ORCA agent buffer (radius + 0.01 per side)
        self.n_directions = 16
        self.n_speeds = 5
        self.time_step = 0.25

    def configure(self, config):
        return

    def set_phase(self, phase):
        return

    def predict(self, state, border=None):
        self_state = state.robot_state
        neighbors = state.human_states
        action = _predict_full_state(
            self_state, neighbors, self.w_a, self.w_b, self.t_min, self.t_max,
            self.neighbor_dist, self.n_directions, self.n_speeds, self.safety_space)
        self.last_state = state
        return action


class CentralizedPLEdestrians(PLEdestrians):
    def predict(self, state, border=None):
        actions = []
        for i, agent_state in enumerate(state):
            neighbors = [s for j, s in enumerate(state) if j != i]
            actions.append(_predict_full_state(
                agent_state, neighbors, self.w_a, self.w_b, self.t_min, self.t_max,
                self.neighbor_dist, self.n_directions, self.n_speeds, self.safety_space))
        return actions
