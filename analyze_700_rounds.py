import numpy as np  # 修改
import matplotlib.pyplot as plt  # 修改
import matplotlib  # 修改
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 修改：支持中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 修改：支持负号

print("="*60)  # 修改
print("700轮训练数据分析")  # 修改
print("="*60)  # 修改

# 读取所有数据  # 修改
try:  # 修改
    actor_losses = np.load('actor_losses.npy')  # 修改
    critic_losses = np.load('critic1_losses.npy')  # 修改

    # 尝试读取 Q 值数据  # 修改
    try:  # 修改
        q_values = np.load('q_values.npy')  # 修改
        target_q_values = np.load('target_q_values.npy')  # 修改
        has_q_data = True  # 修改
    except:  # 修改
        has_q_data = False  # 修改
        print("警告: 没有找到 q_values.npy，可能是旧版本训练数据")  # 修改

    # 尝试读取 rewards  # 修改
    try:  # 修改
        rewards = np.load('rewards.npy')  # 修改
        has_rewards = True  # 修改
    except:  # 修改
        has_rewards = False  # 修改
        print("警告: 没有找到 rewards.npy")  # 修改

    rld_history = np.load('rld_history.npy')  # 修改
    svc_history = np.load('svc_history.npy')  # 修改
    srd_history = np.load('srd_history.npy')  # 修改

    print(f"\n数据统计:")  # 修改
    print(f"- Actor Loss 更新次数: {len(actor_losses)}")  # 修改
    print(f"- Critic Loss 更新次数: {len(critic_losses)}")  # 修改
    if has_q_data:  # 修改
        print(f"- Q Value 记录次数: {len(q_values)}")  # 修改
    if has_rewards:  # 修改
        print(f"- Episode 数量: {len(rewards)}")  # 修改
    print(f"- RLD 记录次数: {len(rld_history)}")  # 修改

    # 分析 Actor Loss 趋势  # 修改
    print(f"\n1. Actor Loss 分析:")  # 修改
    print(f"   初始值: {actor_losses[0]:.2f}")  # 修改
    print(f"   最终值: {actor_losses[-1]:.2f}")  # 修改
    print(f"   最大值: {actor_losses.max():.2f} (第 {actor_losses.argmax()} 次更新)")  # 修改
    print(f"   最小值: {actor_losses.min():.2f} (第 {actor_losses.argmin()} 次更新)")  # 修改

    # 分段分析  # 修改
    n = len(actor_losses)  # 修改
    stage1 = actor_losses[:n//3]  # 前1/3  # 修改
    stage2 = actor_losses[n//3:2*n//3]  # 中1/3  # 修改
    stage3 = actor_losses[2*n//3:]  # 后1/3  # 修改

    print(f"   前1/3平均: {stage1.mean():.2f}")  # 修改
    print(f"   中1/3平均: {stage2.mean():.2f}")  # 修改
    print(f"   后1/3平均: {stage3.mean():.2f}")  # 修改

    if stage3.mean() < stage1.mean():  # 修改
        print(f"   ✓ Actor Loss 整体下降 {stage1.mean() - stage3.mean():.2f}")  # 修改
    else:  # 修改
        print(f"   ✗ Actor Loss 整体上升 {stage3.mean() - stage1.mean():.2f}")  # 修改

    # 分析 Q Value 趋势  # 修改
    if has_q_data:  # 修改
        print(f"\n2. Q Value 分析:")  # 修改
        print(f"   初始值: {q_values[0]:.2f}")  # 修改
        print(f"   最终值: {q_values[-1]:.2f}")  # 修改
        print(f"   最大值: {q_values.max():.2f}")  # 修改
        print(f"   最小值: {q_values.min():.2f}")  # 修改

        q_stage1 = q_values[:n//3]  # 修改
        q_stage2 = q_values[n//3:2*n//3]  # 修改
        q_stage3 = q_values[2*n//3:]  # 修改

        print(f"   前1/3平均: {q_stage1.mean():.2f}")  # 修改
        print(f"   中1/3平均: {q_stage2.mean():.2f}")  # 修改
        print(f"   后1/3平均: {q_stage3.mean():.2f}")  # 修改

        if q_stage3.mean() > q_stage1.mean():  # 修改
            print(f"   ✓ Q Value 整体上升 {q_stage3.mean() - q_stage1.mean():.2f} (策略改进)")  # 修改
        else:  # 修改
            print(f"   ✗ Q Value 整体下降 {q_stage1.mean() - q_stage3.mean():.2f} (策略恶化)")  # 修改

    # 分析 Reward 趋势  # 修改
    if has_rewards:  # 修改
        print(f"\n3. Episode Reward 分析:")  # 修改
        print(f"   初始值: {rewards[0]:.2f}")  # 修改
        print(f"   最终值: {rewards[-1]:.2f}")  # 修改
        print(f"   最大值: {rewards.max():.2f}")  # 修改
        print(f"   最小值: {rewards.min():.2f}")  # 修改

        r_n = len(rewards)  # 修改
        r_stage1 = rewards[:r_n//3]  # 修改
        r_stage2 = rewards[r_n//3:2*r_n//3]  # 修改
        r_stage3 = rewards[2*r_n//3:]  # 修改

        print(f"   前1/3平均: {r_stage1.mean():.2f}")  # 修改
        print(f"   中1/3平均: {r_stage2.mean():.2f}")  # 修改
        print(f"   后1/3平均: {r_stage3.mean():.2f}")  # 修改

        if r_stage3.mean() > r_stage1.mean():  # 修改
            print(f"   ✓ Reward 整体上升 {r_stage3.mean() - r_stage1.mean():.2f} (性能改进)")  # 修改
        else:  # 修改
            print(f"   ✗ Reward 整体下降或无变化 {r_stage1.mean() - r_stage3.mean():.2f}")  # 修改

    # 分析 RLD  # 修改
    print(f"\n4. RLD (调度损失距离) 分析:")  # 修改
    steps_per_ep = len(rld_history) // (len(rewards) if has_rewards else 700)  # 修改
    print(f"   每个episode平均步数: {steps_per_ep}")  # 修改
    print(f"   RLD 平均值: {rld_history.mean():.2f}")  # 修改
    print(f"   RLD 标准差: {rld_history.std():.2f}")  # 修改

    # 综合判断  # 修改
    print(f"\n" + "="*60)  # 修改
    print("综合判断:")  # 修改
    print("="*60)  # 修改

    if has_q_data and has_rewards:  # 修改
        q_improved = q_stage3.mean() > q_stage1.mean()  # 修改
        r_improved = r_stage3.mean() > r_stage1.mean()  # 修改

        if q_improved and r_improved:  # 修改
            print("✓ 训练正常：Q值上升且Reward改进")  # 修改
        elif q_improved and not r_improved:  # 修改
            print("⚠ 问题：Q值上升但Reward没有改进")  # 修改
            print("  可能原因：Critic估计偏差，Q值与实际回报脱节")  # 修改
            print("  建议：降低critic_lr或tau，增加batch_size")  # 修改
        elif not q_improved and r_improved:  # 修改
            print("⚠ 异常：Q值下降但Reward改进")  # 修改
            print("  这种情况很少见，可能是Critic学习滞后")  # 修改
        else:  # 修改
            print("✗ 训练失败：Q值和Reward都没有改进")  # 修改
            print("  建议：检查环境、reward设计、网络架构")  # 修改

    print("="*60)  # 修改

except Exception as e:  # 修改
    print(f"错误: {e}")  # 修改
    import traceback  # 修改
    traceback.print_exc()  # 修改
