# Topology-preserving MPCs
This directory contains various MPCs which seek to preserve some global topological plan. There are 3 separate files: **brne_wind.py**, **brne_wind_cv.py**, **cv_wind.py**. All implementations are built on top of the implementations found in the **vecMPC** directory, and achieve the topological constraints using the `predict_wind`, `winding_reject`, and `winding_distribution` methods.

### brne_wind.py
This MPC performs a trajectory rollout using the BRNE implementation adapted to ComplexityNav. The winding numbers of the returned trajectories are used as the topological constraint.

### brne_wind_cv.py
This MPC performs a constant-velocity pedestrian trajectory rollout and the BRNE implementation adapted to ComplexityNav for the robot. The winding numbers of the returned trajectories are used as the topological constraint.


### cv_wind.py
This MPC performs a constant-velocity trajectory rollout. The winding numbers of the returned trajectories are used as the topological constraint.

## Setup
Used Python 3.8.5 for development. 

## Getting started
To run BRNE in the ComplexityNav simulation environment, each MPC class must be added as a policy. To add the class as a policy, include the following lines in **crowd_nav/policy/policy_factory.py**:
```
from crowd_nav.policy.topology_mpc.<FILE> import <CLASS>

policy_factory['<CONTROLLER_NAME>'] = <CLASS>
```

This allows the user to run BRNE and visualize the result from the command line within the **crowd_nav** directory as follows:

```
python test.py --policy <CONTROLLER_NAME> --model_dir data/output --phase test --visualize --test_case 0
```