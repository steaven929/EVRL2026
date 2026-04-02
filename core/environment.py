# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 19:26:35 2024

@author: TLab
"""


import os
import torch
import numpy as np
import random
import gym

from specific.model import Actor, Critic
from core.algorithm import DDPG, PureDDPG, SAC, TD3
from core.buffer import ReplayBuffer
# =========2026.2.19淇敼锛孧ARL Phase-1锛氬鍏ARL缁勪欢==========
from core.marl_agent import IPPOAgent
# ======淇敼缁撴潫


class NormalizedActions(gym.ActionWrapper):
    ''' 灏哸ction鑼冨洿閲嶅畾鍦╗0.1]涔嬮棿锛岃矊浼间笉闇€瑕?
    '''
    def action(self, action):
        # low_bound   = self.action_space.low
        # upper_bound = self.action_space.high
        # action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        # action = np.clip(action, low_bound, upper_bound)
        return action


# =========2026.2.20淇敼锛孧ARL閫傞厤鍣細鑱斿悎鍔ㄤ綔涓庡眬閮ㄨ娴?=========
class MARLAdapter(gym.Wrapper):
    """Wrap single-agent environment with a MARL interface."""

    def __init__(self, env, n_agents):
        super().__init__(env)
        self.n_agents = int(n_agents)
        self._last_global_state = None

    def reset(self, **kwargs):
        reset_ret = self.env.reset(**kwargs)
        global_state = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        self._last_global_state = np.asarray(global_state)
        return self._last_global_state

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.n_agents, 3)
        if actions.shape != expected_shape:
            raise ValueError(f"MARL actions shape {actions.shape} != {expected_shape}")

        actions = np.clip(actions, 0.0, 1.0)
        flat_action = np.concatenate([actions[:, 0], actions[:, 1], actions[:, 2]], axis=0)

        step_ret = self.env.step(flat_action)
        if len(step_ret) == 5:
            global_state, reward, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        else:
            global_state, reward, done, info = step_ret

        self._last_global_state = np.asarray(global_state)
        return self._last_global_state, float(reward), bool(done), info

    def get_agent_obs(self, agent_id):
        if self._last_global_state is None:
            raise RuntimeError("Call reset() before get_agent_obs().")

        if agent_id < 0 or agent_id >= self.n_agents:
            raise ValueError(f"agent_id {agent_id} out of range [0, {self.n_agents})")

        state = np.asarray(self._last_global_state, dtype=np.float32)
        z = self.n_agents
        expected_min_dim = 5 * z + 1
        if state.shape[0] < expected_min_dim:
            raise ValueError(f"global_state dim {state.shape[0]} < required {expected_min_dim}")

        local_obs = np.array([
            state[agent_id],
            state[z + agent_id],
            state[2 * z + agent_id],
            state[3 * z + agent_id],
            state[4 * z + agent_id],
            state[-1],
        ], dtype=np.float32)
        return np.concatenate([local_obs, np.array([float(agent_id)], dtype=np.float32)], axis=0)
# ======淇敼缁撴潫

    # def reverse_action(self, action):
    #     low_bound   = self.action_space.low
    #     upper_bound = self.action_space.high
    #     action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
    #     action = np.clip(action, low_bound, upper_bound)
    #     return action
    
def all_seed(env,seed = 1):
    ''' 涓囪兘鐨剆eed鍑芥暟
    '''
    env.seed(seed) # env config
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # config for CPU
    torch.cuda.manual_seed(seed) # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
def env_agent_config(cfg):
    env = NormalizedActions(gym.make(cfg['env_name'])) # 瑁呴グaction鍣０
    base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    if hasattr(base_env, 'set_day_sampling'):
        base_env.set_day_sampling(
            same_day_train_test=bool(cfg.get('same_day_train_test', False)),
            fixed_day_index=int(cfg.get('fixed_day_index', 1)),
        )
    if cfg['seed'] !=0:
        all_seed(env,seed=cfg['seed'])
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    state_vocab_size = 64
    if hasattr(env.observation_space, 'nvec'):
        state_vocab_size = int(np.max(env.observation_space.nvec))
    cfg.update({"n_states":n_states,"n_actions":n_actions, "state_vocab_size": state_vocab_size}) # 鏇存柊n_states鍜宯_actions鍒癱fg鍙傛暟涓?
    # =========2026.2.19淇敼锛孧ARL Phase-1锛氭敮鎸乵ulti妯″紡==========
    marl_mode = cfg.get('marl_mode', 'single')  # 榛樿single妯″紡

    if marl_mode == 'multi':
        inferred_agents = n_actions // 3
        if inferred_agents <= 0:
            raise ValueError(f"Invalid action dim for MARL: {n_actions}")
        cfg['n_agents'] = int(cfg.get('n_agents', inferred_agents))
        if cfg['n_agents'] != inferred_agents:
            print(f"[MARL] n_agents from cfg ({cfg['n_agents']}) != inferred ({inferred_agents}), use inferred.")
            cfg['n_agents'] = inferred_agents

        env = MARLAdapter(env, n_agents=cfg['n_agents'])

        # 澶氭櫤鑳戒綋妯″紡锛欼PPO with Parameter Sharing
        agent = IPPOAgent(
            obs_dim=6,  # 灞€閮ㄨ娴嬬淮搴︼紙涓嶅惈agent_id锛?
            n_agents=cfg.get('n_agents', 50),  # 鏅鸿兘浣撴暟閲忥紙鍖哄煙鏁帮級
            action_dim=3,  # 姣忎釜鏅鸿兘浣撶殑鍔ㄤ綔缁村害
            global_state_dim=n_states,  # 鍏ㄥ眬鐘舵€佺淮搴?
            agent_embed_dim=16,  # agent_id embedding缁村害
            actor_hidden_dim=cfg.get('actor_hidden_dim', 128),
            critic_hidden_dim=cfg.get('critic_hidden_dim', 256),
            actor_lr=cfg['actor_lr'],
            critic_lr=cfg['critic_lr'],
            gamma=cfg['gamma'],
            tau=cfg['tau'],
            batch_size=cfg['batch_size'],
            memory_capacity=cfg['memory_capacity'],
            policy_delay=cfg.get('marl_policy_delay', cfg.get('policy_delay', 2)),
            target_policy_noise=cfg.get('marl_target_policy_noise', cfg.get('target_policy_noise', 0.08)),
            target_noise_clip=cfg.get('marl_target_noise_clip', cfg.get('target_noise_clip', 0.15)),
            device=cfg['device']
        )
        return env, agent
    # ======淇敼缁撴潫

    # 鏍规嵁绠楁硶鍚嶇О閫夋嫨涓嶅悓鐨刟gent锛堝崟鏅鸿兘浣撴ā寮忥級
    if cfg['algo_name'] == 'SAC':
        memory = ReplayBuffer(cfg['memory_capacity'])
        # =========2026.2.19淇敼锛屽鍔爒ocab_size浠ユ敮鎸丼IP鏁版嵁闆?=========
        # SIP鏁版嵁闆嗭細OD_MAX=13, TIME_NUM=26
        # vocab_size闇€瑕?>= max(OD_MAX, TIME_NUM) = 26
        agent = SAC(
            n_states=n_states,
            n_actions=n_actions,
            memory=memory,
            vocab_size=30,  # 浠?2澧炲姞鍒?0锛屾敮鎸丼IP鏁版嵁闆嗭紙OD_MAX=13, TIME_NUM=26锛?
            # ======淇敼缁撴潫
            embedding_dims=cfg['actor_embed_dim'],
            hidden_dim=cfg['actor_hidden_dim'],
            use_dual_head=cfg.get('use_dual_head', False),  # 鏄惁浣跨敤鍙屽ご鏋舵瀯
            actor_lr=cfg['actor_lr'],
            critic_lr=cfg['critic_lr'],
            alpha_lr=cfg.get('sac_alpha_lr', 3e-4),
            gamma=cfg['gamma'],
            tau=cfg['tau'],
            alpha=cfg.get('sac_alpha', 0.2),
            batch_size=cfg['batch_size'],
            device=cfg['device']
        )
    elif cfg['algo_name'] == 'TD3':
        models = {
            "actor": Actor(n_states, n_actions, embedding_dims=cfg['actor_embed_dim'], hidden_dim=cfg['actor_hidden_dim'], vocab_size_override=cfg['state_vocab_size']),
            "critic": Critic(n_states, n_actions, embedding_dims=cfg['actor_embed_dim'], hidden_dim=cfg['critic_hidden_dim'], vocab_size_override=cfg['state_vocab_size']),
            "critic2": Critic(n_states, n_actions, embedding_dims=cfg['actor_embed_dim'], hidden_dim=cfg['critic_hidden_dim'], vocab_size_override=cfg['state_vocab_size']),
        }
        memories = {"memory": ReplayBuffer(cfg['memory_capacity'])}
        agent = TD3(models, memories, cfg)
    elif cfg['algo_name'] == 'DDPG_PURE':
        models = {
            "actor": Actor(n_states, n_actions, embedding_dims=cfg['actor_embed_dim'], hidden_dim=cfg['actor_hidden_dim'], vocab_size_override=cfg['state_vocab_size']),
            "critic": Critic(n_states, n_actions, embedding_dims=cfg['actor_embed_dim'], hidden_dim=cfg['critic_hidden_dim'], vocab_size_override=cfg['state_vocab_size']),
        }
        memories = {"memory": ReplayBuffer(cfg['memory_capacity'])}
        agent = PureDDPG(models, memories, cfg)
    else:  # DDPG
        models = {
            "actor": Actor(n_states, n_actions, embedding_dims=cfg['actor_embed_dim'], hidden_dim=cfg['actor_hidden_dim'], vocab_size_override=cfg['state_vocab_size']),
            "critic": Critic(n_states, n_actions, embedding_dims=cfg['actor_embed_dim'], hidden_dim=cfg['critic_hidden_dim'], vocab_size_override=cfg['state_vocab_size']),
        }
        memories = {"memory":ReplayBuffer(cfg['memory_capacity'])}
        agent = DDPG(models,memories,cfg)

    return env,agent

#--
#--------------------5.11
import numpy as np


def manual_action_test(env, custom_action=None):
    print("\n=== 鎵嬪姩鍔ㄤ綔娴嬭瘯寮€濮?===")

    obs = env.reset()

    # 浠庡姩浣滅┖闂存帹鏂?ZONE_NUM
    Z = env.action_space.shape[0] // 3

    # 濡傛灉鏈寚瀹氬姩浣滐紝鍒欐瀯閫犱竴涓湁浠ｈ〃鎬х殑鍔ㄤ綔
    if custom_action is None:
        custom_action = np.array([
            1.0, 0.0, 0.0, 0.0,  # 鍑哄彂鍊惧悜锛氬彧浠庡尯鍩?鍙戣溅
            0.0, 1.0, 0.0, 0.0,  # 鐩殑鍦板€惧悜锛氶€佸幓鍖哄煙1
            1.0, 0.0, 0.0, 0.0  # 鍏呯數鍊惧悜锛氬亸濂藉幓鍖哄煙1鍏呯數
        ])

    print("\n[Manual Action Input]")
    print("鍑哄彂鍊惧悜:", custom_action[0:Z])
    print("鐩殑鍦板€惧悜:", custom_action[Z:2 * Z])
    print("鍏呯數鍊惧悜:", custom_action[2 * Z:3 * Z])

    # 鎵ц涓€姝ヨ皟搴?
    obs, reward, done, _ = env.step(custom_action)

    print(f"\n[Step 0 Completed] Reward = {reward:.2f}, Terminal = {done}")
    print("=== 鎵嬪姩鍔ㄤ綔娴嬭瘯缁撴潫 ===\n")


def test_handle_vehicle_charging(self):
    # 鍋囪杞﹁締缂栧彿涓?
    vehicle_id = 0
    self.car = [[0, 100, 0]]  # 褰撳墠浣嶇疆0锛岀數閲?00锛屽綋鍓嶆棤浠诲姟

    self.dist = np.array([
        [0, 10, 20, 30],
        [10, 0, 15, 25],
        [20, 15, 0, 10],
        [30, 25, 10, 0]
    ])

    self.available = np.array([
        [1, 1, 0, 0],  # 褰撳墠娉婁綅鍗犵敤鎯呭喌
        [2, 2, 2, 2],  # 澶氫綑琛岋細鐢ㄤ簬pre_avail瀵规瘮
    ])

    self.mem_available = np.array([
        [2, 2, 2, 2]  # 鎬诲閲?
    ])

    global CHARGER_ZONES
    CHARGER_ZONES = [1, 2]  # 鍋囪1銆?鍖烘湁鍏呯數妗?

    # case: 鐩爣鍖哄煙 1 鏈夋々鏈変綅锛屽簲鐩存帴娲惧線
    print("馃殫 杞﹁締鍓嶅線鍖哄煙 1")
    result = self.handle_vehicle_charging(vehicle_id, 1)
    print(f"缁撴灉鐩爣鍖哄煙: {result}, 杞﹁締鐘舵€? {self.car[vehicle_id]}")

    # case: 鐩爣鍖哄煙 2 娌℃湁浣嶏紝搴斿鎵炬浛浠ｅ尯
    self.available[0][2] = 2  # 宸叉弧
    print("馃殫 杞﹁締灏濊瘯鍓嶅線宸叉弧鐨勫尯鍩?2")
    result = self.handle_vehicle_charging(vehicle_id, 2)
    print(f"缁撴灉鐩爣鍖哄煙: {result}, 杞﹁締鐘舵€? {self.car[vehicle_id]}")

    # case: 褰撳墠鐢甸噺涓嶈冻浠ュ埌杈剧洰鏍?
    self.car[vehicle_id][1] = 5  # 鐢甸噺杩囦綆
    print("馃殫 鐢甸噺涓嶈冻锛屽墠寰€鍖哄煙 3锛堟棤鍏呯數妗╋級")
    result = self.handle_vehicle_charging(vehicle_id, 3)
    print(f"缁撴灉鐩爣鍖哄煙: {result}, 杞﹁締鐘舵€? {self.car[vehicle_id]}")


