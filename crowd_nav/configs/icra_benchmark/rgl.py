from crowd_nav.configs.icra_benchmark.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig, BaseExperimentsConfig, Config

class ExperimentsConfig(BaseExperimentsConfig):
    def __init__(self, debug=False):
        super(ExperimentsConfig, self).__init__(debug)

class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)
        # Circle-crossing spawn/goals for all train / val / test rollouts
        self.sim.test_scenario = 'circle_crossing'
        self.sim.train_val_scenario = 'circle_crossing'
        # 15 PLEdestrians only (no ORCA / SFM / CV / static)
        self.humans.num_pledestrians = [15]
        self.humans.num_powerlaw = [0]
        self.humans.num_orca = [0]
        self.humans.num_sf = [0]
        self.humans.num_linear = [0]
        self.humans.num_static = [0]


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)
        self.name = 'gcn'

        # gcn
        self.gcn.num_layer = 2
        self.gcn.X_dim = 32
        self.gcn.similarity_function = 'embedded_gaussian'
        self.gcn.layerwise_graph = False
        self.gcn.skip_connection = True


class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)
        self.imitation_learning.il_episodes = 5 
