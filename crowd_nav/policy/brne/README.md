# Mixed Strategy Nash Equilibrium for Crowd Navigation
This directory contains an implementation of the work [Mixed strategy Nash equilibrium for
crowd navigation (BRNE)](https://arxiv.org/pdf/2403.01537) by Sun. et. al. (2024). The code is adapted from Katie Hughes' [ROS2 implementation](https://github.com/katie-hughes/brne_social_nav), and is designed to run as part of the [ComplexityNav](https://github.com/fluentrobotics/ComplexityNav/tree/master) repository. 

## Code overview
The implementation is split up into 3 files: **brne.py**, **brne_driver.py**, and **traj_tracker.py**. **brne.py** contains the core BRNE algorithm and helper functions for computing and updating the weights of a trajectory distribution. For further information on the BRNE algorithm, please refer to the [original paper](https://arxiv.org/pdf/2403.01537). **brne_driver.py** contains the driver class `BRNE_Driver` for running BRNE in the ComplexityNav simulation environment. **traj_tracker.py** contains the `TrajTracker` class which enables rollout (cmds --> states) and tracking (states --> cmds) of trajectories for use in the BRNE algorithm.  

## Setup
Used Python 3.8.5 for development.

## Getting started
To run BRNE in the ComplexityNav simulation environment, `BRNE_Driver` must be added as a policy. To add the class as a policy, include the following lines in **crowd_nav/policy/policy_factory.py**:
```
from crowd_nav.policy.brne.brne_driver import BRNE_Driver

policy_factory['brne'] = BRNE_Driver
```

This allows the user to run BRNE and visualize the result from the command line within the **crowd_nav** directory as follows:

```
python test.py --policy brne --model_dir data/output --phase test --visualize --test_case 0
```