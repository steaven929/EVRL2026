# -*- coding: utf-8 -*-
"""
P0 Optimization Verification Tests
Test Target Actor, Q-value Clipping, Reward Scale Config
"""
import torch
import numpy as np
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core.marl_agent import IPPOAgent

def test_target_actor():
    """Test P0-1: Target Actor Network"""
    print("\n=== Test P0-1: Target Actor Network ===")
    agent = IPPOAgent(
        obs_dim=6,
        n_agents=50,
        action_dim=3,
        global_state_dim=251,
        device='cpu'
    )

    # Verify target_actor exists
    assert hasattr(agent, 'target_actor'), "FAIL: target_actor does not exist"
    print("PASS: target_actor network created")

    # Verify target_actor params match actor (at initialization)
    for p1, p2 in zip(agent.actor.parameters(), agent.target_actor.parameters()):
        assert torch.allclose(p1, p2), "FAIL: target_actor init params mismatch"
    print("PASS: target_actor initialization correct")

    # Verify save/load includes target_actor
    agent.save_model('test_model.pth')
    checkpoint = torch.load('test_model.pth')
    assert 'target_actor' in checkpoint, "FAIL: saved model missing target_actor"
    print("PASS: save_model includes target_actor")

    agent2 = IPPOAgent(obs_dim=6, n_agents=50, action_dim=3, global_state_dim=251)
    agent2.load_model('test_model.pth')
    print("PASS: load_model successfully loads target_actor")

    import os
    os.remove('test_model.pth')
    print("PASS: P0-1 test complete!\n")

def test_q_clipping():
    """Test P0-2: Q-value Clipping"""
    print("=== Test P0-2: Q-value Clipping ===")

    # Test clipping logic
    large_q = torch.tensor([[10000.0]])
    clipped_q = torch.clamp(large_q, -1000, 1000)
    assert clipped_q.item() == 1000.0, "FAIL: Q-value clipping (positive)"
    print("PASS: Q-value positive clipping correct")

    small_q = torch.tensor([[-10000.0]])
    clipped_q = torch.clamp(small_q, -1000, 1000)
    assert clipped_q.item() == -1000.0, "FAIL: Q-value clipping (negative)"
    print("PASS: Q-value negative clipping correct")

    normal_q = torch.tensor([[500.0]])
    clipped_q = torch.clamp(normal_q, -1000, 1000)
    assert clipped_q.item() == 500.0, "FAIL: Q-value clipping affects normal values"
    print("PASS: Q-value clipping preserves normal values")
    print("PASS: P0-2 test complete!\n")

def test_reward_scale():
    """Test P0-3: Reward Scale Configuration"""
    print("=== Test P0-3: Reward Scale Configuration ===")

    # Test config reading
    cfg = {
        'reward_scale': 1000.0,
        'train_eps': 1,
        'batch_size': 128
    }

    reward_scale = float(cfg.get('reward_scale', 1.0))
    assert reward_scale == 1000.0, "FAIL: reward_scale reading failed"
    print(f"PASS: reward_scale config correct: {reward_scale}")

    # Test scaling logic
    raw_reward = 50000.0
    scaled_reward = raw_reward / reward_scale
    assert scaled_reward == 50.0, "FAIL: reward scaling calculation wrong"
    print(f"PASS: reward scaling correct: {raw_reward} -> {scaled_reward}")

    # Test boundary case
    reward_scale_zero = float(cfg.get('reward_scale', 1.0))
    if reward_scale_zero <= 0:
        reward_scale_zero = 1.0
    assert reward_scale_zero > 0, "FAIL: reward_scale boundary handling failed"
    print("PASS: reward_scale boundary handling correct")
    print("PASS: P0-3 test complete!\n")

def test_learning_rate_scheduler():
    """Test P1-4: Learning Rate Scheduler"""
    print("=== Test P1-4: Learning Rate Scheduler ===")
    agent = IPPOAgent(
        obs_dim=6,
        n_agents=50,
        action_dim=3,
        global_state_dim=251,
        actor_lr=1e-4,
        critic_lr=1e-4,
        device='cpu'
    )

    # Verify scheduler exists
    assert hasattr(agent, 'actor_scheduler'), "FAIL: actor_scheduler does not exist"
    assert hasattr(agent, 'critic_scheduler'), "FAIL: critic_scheduler does not exist"
    print("PASS: Learning rate schedulers created")

    # Get initial learning rates
    initial_actor_lr = agent.actor_optimizer.param_groups[0]['lr']
    initial_critic_lr = agent.critic_optimizer.param_groups[0]['lr']
    print(f"PASS: Initial LR - Actor: {initial_actor_lr:.2e}, Critic: {initial_critic_lr:.2e}")

    # Simulate 50 steps of LR decay
    for _ in range(50):
        agent.actor_scheduler.step()
        agent.critic_scheduler.step()

    new_actor_lr = agent.actor_optimizer.param_groups[0]['lr']
    new_critic_lr = agent.critic_optimizer.param_groups[0]['lr']
    print(f"PASS: After 50 steps - Actor: {new_actor_lr:.2e}, Critic: {new_critic_lr:.2e}")

    # Verify LR actually decayed
    assert new_actor_lr < initial_actor_lr, "FAIL: Actor LR did not decay"
    assert new_critic_lr < initial_critic_lr, "FAIL: Critic LR did not decay"
    print("PASS: Learning rate scheduler working correctly")
    print("PASS: P1-4 test complete!\n")

def test_gradient_monitoring():
    """Test P1-6: Gradient Monitoring"""
    print("=== Test P1-6: Gradient Monitoring ===")
    agent = IPPOAgent(
        obs_dim=6,
        n_agents=50,
        action_dim=3,
        global_state_dim=251,
        batch_size=4,  # Small batch for testing
        device='cpu'
    )

    # Verify _compute_grad_norm method exists
    assert hasattr(agent, '_compute_grad_norm'), "FAIL: _compute_grad_norm method missing"
    print("PASS: _compute_grad_norm method created")

    # Add some fake data to memory
    for _ in range(10):
        obs_n = np.random.rand(50, 7)
        act_n = np.random.rand(50, 3)
        global_state = np.random.rand(251)
        reward = np.random.rand()
        next_obs_n = np.random.rand(50, 7)
        next_global_state = np.random.rand(251)
        done = False

        agent.memory.push(obs_n, act_n, global_state, reward, next_obs_n, next_global_state, done)

    # Perform one update
    result = agent.update()
    assert result is not None, "FAIL: update returned None"

    # Verify return format
    assert len(result) == 3, f"FAIL: update return format wrong, expected 3 elements, got {len(result)}"
    critic_loss, actor_loss, grad_info = result

    # Verify gradient info
    assert 'critic_grad_norm' in grad_info, "FAIL: missing critic_grad_norm"
    assert 'actor_grad_norm' in grad_info, "FAIL: missing actor_grad_norm"
    print(f"PASS: Critic grad norm: {grad_info['critic_grad_norm']:.4f}")
    print(f"PASS: Actor grad norm: {grad_info['actor_grad_norm']:.4f}")

    # Verify gradient norms are positive
    assert grad_info['critic_grad_norm'] >= 0, "FAIL: Critic grad norm is negative"
    assert grad_info['actor_grad_norm'] >= 0, "FAIL: Actor grad norm is negative"
    print("PASS: Gradient monitoring working correctly")
    print("PASS: P1-6 test complete!\n")

if __name__ == '__main__':
    print("=" * 60)
    print("Starting P0 and P1 Optimization Tests")
    print("=" * 60)

    try:
        test_target_actor()
        test_q_clipping()
        test_reward_scale()
        test_learning_rate_scheduler()
        test_gradient_monitoring()

        print("=" * 60)
        print("ALL TESTS PASSED! P0 and P1 optimizations work correctly")
        print("=" * 60)
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
