import gym
from gym.envs.registration import register
import numpy as np
import os
import shutil

# Mock config if needed, but dclily_elec loads from json
# We need to make sure we are in the right directory or config is accessible
# dclily_elec.py expects specific/config.json relative to script_dir

# Register env
register(
    id='dclily-v0',
    entry_point='specific.dclily_elec:dclily',
)

def verify():
    print("Initializing environment...")
    try:
        env = gym.make('dclily-v0')
    except Exception as e:
        print(f"Failed to make env: {e}")
        # Try to fix path issue if config not found?
        # dclily_elec.py: script_dir = os.path.dirname(os.path.abspath('...'))
        # It uses absolute path in the code I saw! 
        # 'F:\\AAAA组会\\代码备份\\2501007加了conj参数后备份\\dclily_torch_ddpg_demo基于0425修改\\specific'
        # So it should work if that path is correct on this machine.
        return

    print("Resetting environment...")
    env.reset()
    
    print("Stepping environment...")
    action = env.action_space.sample()
    # Mock action to ensure it has correct shape/values if needed
    # Action space is Box(0, 1, (3*ZONE_NUM,))
    
    env.step(action)
    
    print("Checking for log file...")
    if os.path.exists("vehicle_action_log.csv"):
        print("Log file exists!")
        with open("vehicle_action_log.csv", "r") as f:
            lines = f.readlines()
            print(f"Log file has {len(lines)} lines.")
            print("First 5 lines:")
            for line in lines[:5]:
                print(line.strip())
    else:
        print("Log file NOT found.")

if __name__ == "__main__":
    verify()
