import inspect

from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, ob, border=None, baseline=None):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')

        state = JointState(self.get_full_state(), ob)
        params = inspect.signature(self.policy.predict).parameters
        if 'border' in params:
            kwargs = {'border': border, 'baseline': baseline}
            if 'radius' in params:
                kwargs['radius'] = None
            action = self.policy.predict(state, **kwargs)
        else:
            action = self.policy.predict(state)
        return action