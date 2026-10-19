# EVRL2026 本机环境（2026-10-15）

已创建独立 Python 3.12.6 环境，未修改全局 Python 包。

- 解释器：`C:\liuchengjie\EVRL2026\.venv\Scripts\python.exe`
- PyCharm 项目解释器名称：`Python 3.12 (EVRL2026)`
- 入口：`main.py`；工作目录：`C:\liuchengjie\EVRL2026`
- 已修正 `.idea` 的项目解释器和 main 运行配置，已在本机 PyCharm Community 2024.2 注册环境。
- 重新打开项目以加载配置。其他版本 PyCharm：Settings > Project > Python Interpreter > Add Interpreter > Add Local Interpreter，选择上述现有解释器。
- PyTorch 2.4.1 为 CPU 版，匹配代码 `device=cpu` 默认设置。

## Gurobi

原许可证已于 2026-10-01 到期。
main 运行配置的环境变量 `GRB_LICENSE_FILE` 已切换到新许可证：
`C:\gurobi1300\win64\bin\gurobi.lic`
该许可证为本机 Gurobi 13 正式许可证文件，已通过 3000 个二进制变量以及 EVRL2026 的 200 车、50 区域 P0 实际求解验证。
已同时设置用户环境变量 `GRB_LICENSE_FILE`、`GUROBI_HOME` 和 Gurobi `bin` 路径；重启 PyCharm 后，其他运行配置也可自动继承。
后续扩大求解规模或到期时，请更新正式/学术许可证，并相应修改此环境变量。

## 验证

pip check 无依赖冲突，所有第三方依赖均可导入。
在临时副本中运行 main.py，临时覆盖 train_eps=1、test_eps=1、batch_size=4，完成训练、实际梯度更新、Gurobi 求解、模型保存、绘图及测试。
原项目代码、模型、训练数据和既有结果没有被试跑覆盖。
Gym 停止维护提示，以及 pandas/绘图库的弃用警告，不影响本次验证运行。

## 依赖与恢复

直接依赖见 requirements.txt，当前完整版本见 requirements-lock.txt。
重建时使用此环境的 python.exe 执行 `-m pip install -r requirements-lock.txt`。
修改前的 PyCharm XML 配置旁已保存 `.before-evrl-env.bak` 备份。





