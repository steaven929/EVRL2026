import numpy as np  # 修改
import matplotlib.pyplot as plt  # 修改

# 读取最近一次训练的数据  # 修改
try:  # 修改
    # 尝试读取 rewards（如果有保存的话）  # 修改
    # 注意：main.py 中没有保存 rewards.npy，所以这里可能会失败  # 修改
    print("正在查找训练数据...")  # 修改

    # 检查是否有 actor_losses  # 修改
    actor_losses = np.load('actor_losses.npy')  # 修改
    print(f"Actor losses 数量: {len(actor_losses)}")  # 修改
    print(f"Actor losses 范围: {actor_losses.min():.2f} 到 {actor_losses.max():.2f}")  # 修改

    # 检查 RLD  # 修改
    rld = np.load('rld_history.npy')  # 修改
    print(f"\nRLD 数量: {len(rld)}")  # 修改
    print(f"RLD 前10个: {rld[:10]}")  # 修改
    print(f"RLD 后10个: {rld[-10:]}")  # 修改

    # 计算每个 episode 的平均 RLD  # 修改
    # 假设每个 episode 有 max_steps 步  # 修改
    # 从数据量推断：3600 个 RLD 数据点  # 修改
    print(f"\n如果是300个episode，每个episode平均 {len(rld)/300:.1f} 步")  # 修改

    # 尝试按 episode 分组计算平均 RLD  # 修改
    steps_per_episode = 12  # 假设每个episode 12步  # 修改
    num_episodes = len(rld) // steps_per_episode  # 修改
    episode_rld = []  # 修改
    for i in range(num_episodes):  # 修改
        start = i * steps_per_episode  # 修改
        end = start + steps_per_episode  # 修改
        episode_rld.append(np.mean(rld[start:end]))  # 修改

    print(f"\n按 episode 计算的平均 RLD:")  # 修改
    print(f"前10个episode: {episode_rld[:10]}")  # 修改
    print(f"后10个episode: {episode_rld[-10:]}")  # 修改

    # 绘制趋势  # 修改
    plt.figure(figsize=(12, 4))  # 修改
    plt.subplot(1, 2, 1)  # 修改
    plt.plot(rld, alpha=0.3, label='Step-wise RLD')  # 修改
    plt.xlabel('Step')  # 修改
    plt.ylabel('RLD')  # 修改
    plt.title('RLD per Step')  # 修改
    plt.legend()  # 修改
    plt.grid(True, alpha=0.3)  # 修改

    plt.subplot(1, 2, 2)  # 修改
    plt.plot(episode_rld, label='Episode Average RLD')  # 修改
    plt.xlabel('Episode')  # 修改
    plt.ylabel('Average RLD')  # 修改
    plt.title('RLD per Episode')  # 修改
    plt.legend()  # 修改
    plt.grid(True, alpha=0.3)  # 修改

    plt.tight_layout()  # 修改
    plt.savefig('rld_analysis.png', dpi=150)  # 修改
    print("\n已保存 RLD 分析图: rld_analysis.png")  # 修改

except FileNotFoundError as e:  # 修改
    print(f"文件未找到: {e}")  # 修改
except Exception as e:  # 修改
    print(f"错误: {e}")  # 修改

print("\n" + "="*50)  # 修改
print("诊断说明:")  # 修改
print("1. 如果 RLD 在下降，说明调度效率在提升")  # 修改
print("2. 如果 episode 平均 RLD 没有明显趋势，可能是:")  # 修改
print("   - 环境本身的随机性很大")  # 修改
print("   - 策略改进不明显")  # 修改
print("   - 需要更多训练轮次")  # 修改
print("="*50)  # 修改
