import os


def build_outputs_dir(prefix, train_eps, curr_time, scenario_tag="1000car"):
    return f"./outputs/{prefix}_{scenario_tag}_{int(train_eps)}ep_{curr_time}/"


def build_single_agent_run_dir(output_root, run_stamp, algo_name, train_eps, scenario_tag="1000car"):
    dirname = f"{run_stamp}_{str(algo_name).upper()}_{scenario_tag}_{int(train_eps)}ep"
    return os.path.join(output_root, dirname)
