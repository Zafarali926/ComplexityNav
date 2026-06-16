import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY

_EPSILON = 1e-5


def _compute_power_law_force(px, py, vx, vy, radius, opx, opy, ovx, ovy, oradius,
                             k=1.5, tau0=3.0, safety_space=0.01, tau_min=0.05):
    """Interaction force from Karamouzas et al. (2014), ported from UMANS PowerLaw.cpp."""
    R = oradius + radius + 2.0 * safety_space
    x = np.array([px - opx, py - opy], dtype=np.float64)
    v = np.array([vx - ovx, vy - ovy], dtype=np.float64)
    a = np.dot(v, v)
    b = -np.dot(x, v)
    c = np.dot(x, x) - R * R

    if c <= 0:
        return np.zeros(2)

    d = b * b - a * c
    if d <= 0 or a < _EPSILON:
        return np.zeros(2)

    sqrt_d = np.sqrt(d)
    tau = (b - sqrt_d) / a
    if tau < 0.001 or tau > tau0:
        return np.zeros(2)

    # Keep TTC gating unchanged, but regularize denominator near tau -> 0 to avoid spikes.
    tau_eff = max(tau, tau_min)
    component1 = -k * np.exp(-tau / tau0) * (2.0 / tau_eff + 1.0 / tau0) / (a * tau_eff * tau_eff)
    component2 = v - (a * x + b * v) / sqrt_d
    force = component1 * component2
    # Smoothly saturate very large forces near singular TTC to reduce frame-to-frame jitter.
    max_force = 10.0
    mag = np.linalg.norm(force)
    if mag > _EPSILON:
        force = force * (np.tanh(mag / max_force) * max_force / mag)
    return force


def _goal_force(px, py, gx, gy, vx, vy, v_pref, tau_g=0.5):
    """Drive agent toward its goal at preferred speed (UMANS GoalReachingForce analogue)."""
    direction = np.array([gx - px, gy - py], dtype=np.float64)
    dist = np.linalg.norm(direction)
    if dist < _EPSILON:
        return np.zeros(2)
    desired = direction / dist * v_pref
    current = np.array([vx, vy], dtype=np.float64)
    return (desired - current) / tau_g


def _clip_velocity(vx, vy, max_speed):
    speed = np.hypot(vx, vy)
    if speed < _EPSILON:
        return 0.0, 0.0
    if speed > max_speed:
        scale = max_speed / speed
        return vx * scale, vy * scale
    return vx, vy


def _predict_full_state(agent_state, neighbors, k, tau0, neighbor_dist, time_step, tau_g, safety_space,
                        vel_blend=0.75, tau_min=0.05):
    total_force = _goal_force(
        agent_state.px, agent_state.py, agent_state.gx, agent_state.gy,
        agent_state.vx, agent_state.vy, agent_state.v_pref, tau_g=tau_g)

    for other in neighbors:
        dist = np.hypot(agent_state.px - other.px, agent_state.py - other.py)
        if dist > neighbor_dist:
            continue
        total_force += _compute_power_law_force(
            agent_state.px, agent_state.py, agent_state.vx, agent_state.vy, agent_state.radius,
            other.px, other.py, other.vx, other.vy, other.radius,
            k=k, tau0=tau0, safety_space=safety_space, tau_min=tau_min)

    vx_raw = agent_state.vx + total_force[0] * time_step
    vy_raw = agent_state.vy + total_force[1] * time_step
    # Velocity blending damps frame-to-frame switching in close interactions.
    vx = vel_blend * vx_raw + (1.0 - vel_blend) * agent_state.vx
    vy = vel_blend * vy_raw + (1.0 - vel_blend) * agent_state.vy
    vx, vy = _clip_velocity(vx, vy, agent_state.v_pref)
    return ActionXY(vx, vy)


class PowerLaw(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'powerlaw'
        self.trainable = False
        self.multiagent_training = True
        self.kinematics = 'holonomic'
        self.k = 1.5
        self.tau0 = 3.0
        self.tau_g = 0.8
        self.neighbor_dist = 10.0
        self.safety_space = 0.01  # matches ORCA agent buffer (radius + 0.01 per side)
        self.tau_min = 0.05
        self.vel_blend = 0.75
        # Smaller integration step improves stability while preserving Power Law dynamics.
        self.time_step = 0.1

    def configure(self, config):
        return

    def set_phase(self, phase):
        return

    def predict(self, state, border=None):
        self_state = state.robot_state
        neighbors = state.human_states
        action = _predict_full_state(
            self_state, neighbors, self.k, self.tau0, self.neighbor_dist, self.time_step, self.tau_g,
            self.safety_space, self.vel_blend, self.tau_min)
        self.last_state = state
        return action


class CentralizedPowerLaw(PowerLaw):
    def predict(self, state, border=None):
        actions = []
        for i, agent_state in enumerate(state):
            neighbors = [s for j, s in enumerate(state) if j != i]
            actions.append(_predict_full_state(
                agent_state, neighbors, self.k, self.tau0, self.neighbor_dist, self.time_step, self.tau_g,
                self.safety_space, self.vel_blend, self.tau_min))
        return actions
