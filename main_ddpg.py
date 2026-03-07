# -*- coding: utf-8 -*-
"""Run a pure (vanilla) DDPG baseline via main.py."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Pure DDPG baseline launcher')
    parser.add_argument('--seed', type=int, default=7, help='Random seed')
    parser.add_argument('--train_eps', type=int, default=1000, help='Training episodes (conservative baseline default)')
    parser.add_argument('--test_eps', type=int, default=10, help='Test episodes')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--actor_lr', type=float, default=1e-5, help='Actor learning rate')
    parser.add_argument('--critic_lr', type=float, default=2e-5, help='Critic learning rate')
    parser.add_argument('--tau', type=float, default=1e-4, help='Soft update tau')
    parser.add_argument('--gamma', type=float, default=0.97, help='Discount factor')
    parser.add_argument('--reward_scale', type=float, default=10000.0, help='Reward scaling')
    parser.add_argument('--eval_every', type=int, default=100, help='Periodic eval frequency')
    parser.add_argument('--eval_eps', type=int, default=3, help='Eval episodes each time')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--data_folder', type=str, default='data\\SIP\\', help='Dataset folder passed to main.py')
    return parser.parse_args()


def ensure_sip_config(root: Path, data_folder: str):
    folder = (root / data_folder).resolve()
    cfg_json = folder / 'config.json'
    cfg_sip = folder / 'config_sip.json'
    if cfg_json.exists() or not cfg_sip.exists():
        return
    shutil.copy2(cfg_sip, cfg_json)
    print(f'Prepared SIP config: {cfg_json}')


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    main_py = root / 'main.py'
    ensure_sip_config(root, args.data_folder)

    cmd = [
        sys.executable,
        str(main_py),
        '--algo_name', 'DDPG_PURE',
        '--seed', str(args.seed),
        '--train_eps', str(args.train_eps),
        '--test_eps', str(args.test_eps),
        '--batch_size', str(args.batch_size),
        '--actor_lr', str(args.actor_lr),
        '--critic_lr', str(args.critic_lr),
        '--tau', str(args.tau),
        '--gamma', str(args.gamma),
        '--reward_scale', str(args.reward_scale),
        '--eval_every', str(args.eval_every),
        '--eval_eps', str(args.eval_eps),
        '--device', str(args.device),
        '--data_folder', str(args.data_folder),
    ]

    print('Running pure DDPG baseline:')
    print(' '.join(cmd))
    proc = subprocess.run(cmd, cwd=str(root))
    raise SystemExit(proc.returncode)


if __name__ == '__main__':
    main()
