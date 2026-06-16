from crowd_sim.envs.policy.linear import Linear
from crowd_sim.envs.policy.orca import ORCA, CentralizedORCA
from crowd_sim.envs.policy.socialforce import SocialForce, CentralizedSocialForce
from crowd_sim.envs.policy.powerlaw import PowerLaw, CentralizedPowerLaw
from crowd_sim.envs.policy.pledestrians import PLEdestrians, CentralizedPLEdestrians
from crowd_sim.envs.policy.multi_policy import CentralizedMultiPolicy


def none_policy():
    return None


policy_factory = dict()
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
policy_factory['socialforce'] = SocialForce
policy_factory['powerlaw'] = PowerLaw
policy_factory['pledestrians'] = PLEdestrians
policy_factory['centralized_orca'] = CentralizedORCA
policy_factory['centralized_socialforce'] = CentralizedSocialForce
policy_factory['centralized_powerlaw'] = CentralizedPowerLaw
policy_factory['centralized_pledestrians'] = CentralizedPLEdestrians
policy_factory['centralized_multipolicy'] = CentralizedMultiPolicy
policy_factory['none'] = none_policy
