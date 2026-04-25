# -*- coding: utf-8 -*-
"""Rule-gap-plus-v2 baseline entrypoint with MARL-compatible artifact output."""

import datetime
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gym.envs.registration import register, registry

from core.util import get_args
from main_marl_nooptim import save_training_artifacts
from main_rule_gap_plus import run_rule_gap_plus, run_rule_gap_plus_test


RULE_GAP_PLUS_V2_ENV_ID = "dclily_rule_gap_plus_v2"
RULE_GAP_PLUS_V2_ENTRY_POINT = "specific.dclily_elec_rule_gap_plus_v2:dclily"


def register_rule_gap_plus_v2_env():
    if RULE_GAP_PLUS_V2_ENV_ID not in registry:
        register(id=RULE_GAP_PLUS_V2_ENV_ID, entry_point=RULE_GAP_PLUS_V2_ENTRY_POINT)


def build_cfg():
    cfg = get_args()
    cfg["env_name"] = RULE_GAP_PLUS_V2_ENV_ID
    cfg["algo_name"] = "RULE_GAP_PLUS_V2"
    cfg["marl_mode"] = "single"
    return cfg


def build_output_path(train_eps, timestamp):
    return f"./outputs/rule_gap_plus_v2_{train_eps}ep_{timestamp}/"


def main():
    register_rule_gap_plus_v2_env()
    cfg = build_cfg()

    import gym

    env = gym.make(cfg["env_name"])
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = build_output_path(cfg.get("train_eps", 0), curr_time)
    os.makedirs(path, exist_ok=True)

    res_dic = run_rule_gap_plus(env, cfg)
    save_training_artifacts(path, res_dic, cfg)
    run_rule_gap_plus_test(env, cfg, path=path)
    env.close()
    return res_dic


if __name__ == "__main__":
    main()
