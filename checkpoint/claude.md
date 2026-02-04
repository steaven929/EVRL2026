# Claude Code Rules (2026.02.04 Refined)

## 1. 极简输出规范 (Critical)
- **Silent Mode**: 修改代码时，禁止输出代码块。
- **Info Only**: 仅输出：[文件路径]、[函数/类名]、[修改数学逻辑/原因]。
- **Manual Trigger**: 除非我明确说 "show me the code"，否则保持静默。
- **Context Alert**: 接近 Token 上限前，必须主动提示执行 `/compact`。

## 2. 代码修改强制规范
- **Python 注释**: 涉及 .py 修改，必须使用以下块标记：
  - 起：`# =========2026.1.30修改，最优解求解==========`
  - 止：`# ======修改结束`
- **保护逻辑**: 严禁触碰 Gurobi 求解器核心代码；修改 Reward 公式前必须二次确认。

## 3. 文件管理
- **最小化原则**: 优先修改现有文件，单次任务最多新建 1 个 .py。
- **文档登记**: 新建文件必须在 `checkpoint/readai_me.txt` 登记用途和调用关系。

## 4. 算法核心一致性 (RL vs Optimal Control)
- **Reward 对齐**: 
  - RL Reward = (服务+充电-低电量惩罚-调度成本)/10 - 500
  - 必须检查符号反转（Maximization vs Minimization）。
  - 离散化必须核对时间步长 $\Delta t$。