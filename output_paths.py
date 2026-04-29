import os


def build_main_run_dir(current_dir, algo_name, train_eps, run_stamp):
    output_root = os.path.join(current_dir, 'outputs')
    run_dir = os.path.join(
        output_root,
        f"{run_stamp}_{str(algo_name).upper()}_200car_{int(train_eps)}ep",
    )
    return run_dir


def build_marl_run_dir(train_eps, curr_time):
    return f"./outputs/marl_200car_{int(train_eps)}ep_{curr_time}/"
