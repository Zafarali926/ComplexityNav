#!/usr/bin/env python
"""
Run CrowdSim with humans only (robot hidden, zero motion) to verify paper policy mixtures.

Paper mixes (15 humans each):
  sfm_only  — 15 SFM
  orca_only — 15 ORCA
  mix1      — 8 ORCA + 7 SFM
  mix2      — 5 ORCA + 5 SFM + 2 CV + 3 static
  mix3      — 4 ORCA + 4 SFM + 4 CV + 3 static
  PL_only   — 15 Power Law (circle_crossing spawn)
  PLD_only  — 15 PLEdestrians (circle_crossing spawn)

Examples:
  cd crowd_nav
  python utils/test_human_mixes.py --list
  python utils/test_human_mixes.py --all
  python utils/test_human_mixes.py --human_mix mix2 --steps 80
  python utils/test_human_mixes.py --human_mix mix3 --visualize
"""

import argparse
import importlib.util
import logging
import sys
import os

import gym
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crowd_nav.human_mix_presets import HUMAN_MIX_PRESETS, PAPER_MIX_ORDER, mix_description
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.policy.socialforce import SocialForce
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.powerlaw import PowerLaw
from crowd_sim.envs.policy.pledestrians import PLEdestrians
from crowd_sim.envs.policy.linear import Linear

CIRCLE_CROSSING_MIXES = {'PL_only', 'PLD_only'}


def policy_label(human):
    if isinstance(human.policy, SocialForce):
        return 'SFM'
    if isinstance(human.policy, ORCA):
        return 'ORCA'
    if isinstance(human.policy, PowerLaw):
        return 'PL'
    if isinstance(human.policy, PLEdestrians):
        return 'PLD'
    if isinstance(human.policy, Linear):
        return 'static' if human.v_pref < 0.01 else 'CV'
    return type(human.policy).__name__


def apply_mix(env_config, mix):
    env_config.humans.num_orca = [mix['orca']]
    env_config.humans.num_sf = [mix['sf']]
    env_config.humans.num_powerlaw = [mix.get('powerlaw', 0)]
    env_config.humans.num_pledestrians = [mix.get('pledestrians', 0)]
    env_config.humans.num_linear = [mix['linear']]
    env_config.humans.num_static = [mix['static']]


def load_env_config(config_path):
    spec = importlib.util.spec_from_file_location('config', config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, 'EnvConfig'):
        return mod.EnvConfig(False)
    from crowd_nav.configs.icra_benchmark.config import BaseEnvConfig
    return BaseEnvConfig(False)


def run_mix(name, args):
    mix = HUMAN_MIX_PRESETS[name]
    cfg = load_env_config(args.config)
    apply_mix(cfg, mix)
    if name in CIRCLE_CROSSING_MIXES:
        cfg.sim.test_scenario = 'circle_crossing'

    env = gym.make('CrowdSim-v0')
    env.configure(cfg)
    robot = Robot(cfg, 'robot')
    env.set_robot(robot)
    robot.time_step = env.time_step
    robot.set_policy(policy_factory['orca']())
    robot.policy.time_step = env.time_step
    robot.visible = False
    env.unwrapped.reset(phase='test', test_case=args.test_case)

    hold = ActionXY(0.0, 0.0)
    done = False
    while not done and env.unwrapped.global_time < args.steps:
        _, _, done, _ = env.unwrapped.step(hold)

    total = sum(mix.values())
    print(f'\n=== {name}: {mix_description(name)} ({total} humans, {env.unwrapped.global_time:.2f}s) ===')
    ok = True
    start_idx = 1 if len(env.unwrapped.states) > 1 else 0
    by_type = {}
    for i, human in enumerate(env.unwrapped.humans):
        label = policy_label(human)
        p0 = np.array(env.unwrapped.states[start_idx][1][i].position)
        p1 = np.array(env.unwrapped.states[-1][1][i].position)
        disp = float(np.linalg.norm(p1 - p0))
        expect_move = label != 'static'
        status = 'OK' if (disp > 0.05) == expect_move else 'CHECK'
        if status == 'CHECK':
            ok = False
        by_type.setdefault(label, []).append(disp)
        print(f'  [{i}] {label:6s}  displacement={disp:.3f} m  [{status}]')
    for label, disps in sorted(by_type.items()):
        print(f'  summary {label}: n={len(disps)}, mean_disp={np.mean(disps):.3f} m')
    print('  mix:', 'PASS' if ok else 'FAIL (see CHECK rows)')
    return env, ok


def main():
    parser = argparse.ArgumentParser(description='Human-only paper policy mixture tests')
    parser.add_argument('--config', default='configs/icra_benchmark/config.py')
    parser.add_argument('--human_mix', type=str, default=None,
                        help='One paper preset: sfm_only, orca_only, mix1, mix2, mix3, PL_only, PLD_only')
    parser.add_argument('--all', action='store_true',
                        help='Run all paper mixtures in order')
    parser.add_argument('--list', action='store_true', help='List paper presets and exit')
    parser.add_argument('--steps', type=int, default=80,
                        help='Max sim time (s); time_limit may end earlier')
    parser.add_argument('--test_case', type=int, default=0)
    parser.add_argument('--visualize', action='store_true',
                        help='Show matplotlib animation after the run')
    parser.add_argument('--video_file', type=str, default=None)
    args = parser.parse_args()

    if args.list:
        print('Paper policy mixtures (15 humans each):')
        for k in PAPER_MIX_ORDER:
            print(f'  {k:10s}  {mix_description(k)}')
        return

    logging.basicConfig(level=logging.WARNING)
    names = PAPER_MIX_ORDER if args.all else [args.human_mix]
    if not args.all and not args.human_mix:
        parser.error('Specify --human_mix NAME or --all. Use --list for choices.')
    if args.human_mix and args.human_mix not in HUMAN_MIX_PRESETS:
        parser.error(f'Unknown mix {args.human_mix!r}. Use --list.')

    last_env = None
    all_ok = True
    for name in names:
        last_env, ok = run_mix(name, args)
        all_ok = all_ok and ok
        if args.visualize and not args.all:
            if args.video_file:
                last_env.unwrapped.render('video', args.video_file)
            else:
                last_env.unwrapped.render('video', None)

    if args.all:
        print('\nAll paper mixes finished. Re-run one with --visualize to inspect motion.')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
