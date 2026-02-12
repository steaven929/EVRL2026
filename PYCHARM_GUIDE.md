# PyCharm 配置指南 - EVRL2026 项目

## 📋 项目信息

**项目路径:** `C:\Users\cheng\PycharmProjects\EVRL2026`
**Python 解释器:** `C:\Users\cheng\AppData\Local\Programs\Python\Python310\python.exe`
**Python 版本:** 3.10.11

---

## 🚀 快速开始

### 方法 1: 使用启动脚本（推荐）

运行启动脚本：
```cmd
C:\Users\cheng\open-evrl-in-pycharm.bat
```

这将自动打开 PyCharm 并加载 EVRL2026 项目。

---

## 🛠️ PyCharm 配置步骤

### 第 1 步：配置 Python 解释器

1. 打开 PyCharm
2. 进入 **Settings**: `File` → `Settings` (或 `Ctrl + Alt + S`)
3. 导航到: `Project` → `Python Interpreter`
4. 点击 **"Add Interpreter"** (右上角的齿轮图标)
5. 选择 **"System Interpreter"**
6. 点击 **"..."** 浏览 Python 可执行文件
7. 导航到: `C:\Users\cheng\AppData\Local\Programs\Python\Python310\python.exe`
8. 选择 `python.exe` 并点击 **OK**
9. 点击 **OK** 关闭设置窗口

### 第 2 步：安装项目依赖

PyCharm 会自动检测依赖，但你可以手动安装：

**在 PyCharm 终端中：**
1. 打开 PyCharm 终端：`View` → `Tool Windows` → `Terminal` (或 `Alt + F12`)
2. 确保在项目根目录 (`EVRL2026`)
3. 运行：
```bash
pip install -r requirements.txt
```

**或逐个安装：**
```bash
pip install matplotlib gym==0.21.0 torch==2.0.1 torchvision==0.15.2 seaborn pandas scikit-learn
```

### 第 3 步：运行项目

**方法 A: 使用右键菜单**
1. 在 Project 面板中找到 `main.py`
2. 右键点击 `main.py`
3. 选择 **"Run 'main'"**
4. 程序将开始运行

**方法 B: 使用运行配置**
1. 右键点击 `main.py`
2. 选择 **"Modify Run Configuration"**
3. 检查配置：
   - **Script path**: `main.py`
   - **Python interpreter**: 已配置的解释器
   - **Working directory**: `$PROJECT_DIR$`
4. 点击 **OK**
5. 点击绿色的 **Run** 按钮 (或按 `Shift + F10`)

---

## 📊 预期输出

运行 `main.py` 后，你应该看到：

```
训练参数如下：
================================================================================
        参数名         	        参数值         	        参数类型
     algo_name      	        DDPG        	   <class 'str'>
     ...
================================================================================
开始训练！
回合：1/10，奖励：-5325.00
回合：2/10，奖励：-5470.00
...
完成训练！
Loss数据和指标已保存
模型已保存到: trained_model.pth
综合指标图已保存: final_metrics.png
电池热力图已保存: battery_heatmap.png
开始测试！
...
完成测试！
```

---

## 🔧 故障排除

### 问题 1: "ModuleNotFoundError: No module named 'xxx'"

**解决方案:**
```bash
pip install matplotlib numpy gym torch torchvision seaborn pandas scikit-learn
```

### 问题 2: "UnicodeEncodeError: 'charmap' codec can't encode characters"

**解决方案:**
确保运行时设置了正确的编码：
```bash
set PYTHONIOENCODING=utf-8
python main.py
```

### 问题 3: PyCharm 无法找到 Python 解释器

**解决方案:**
1. 检查 Python 是否真的安装在该位置
2. 如果路径不同，手动指定正确的路径
3. 重新启动 PyCharm

### 问题 4: Visual C++ Redistributable 错误

**解决方案:**
PyTorch 需要 Microsoft Visual C++ Redistributable:
1. 下载: `https://aka.ms/vs/17/release/vc_redist.x64.exe`
2. 安装并重启计算机

---

## 📁 生成的文件

运行后，项目目录将包含：

| 文件 | 说明 |
|------|------|
| `trained_model.pth` | 训练好的模型 |
| `critic1_losses.npy` | Critic 损失数据 |
| `rewards.npy` | 回合奖励 |
| `rld_history.npy` | RLD 历史记录 |
| `svc_history.npy` | SVC 历史记录 |
| `srd_history.npy` | SRD 历史记录 |
| `final_metrics.png` | 综合指标图 |
| `battery_heatmap.png` | 电池热力图 |

---

## ⌨️ 快捷键

| 操作 | 快捷键 |
|------|--------|
| 运行项目 | `Shift + F10` |
| 调试项目 | `Shift + F9` |
| 停止运行 | `Ctrl + F2` |
| 打开终端 | `Alt + F12` |
| 打开设置 | `Ctrl + Alt + S` |

---

## 📝 常见任务

### 重新训练模型

1. 在 PyCharm 中打开 `main.py`
2. 修改训练参数（如 `train_eps`, `max_steps` 等）
3. 右键 `main.py` → "Run 'main'"

### 测试训练好的模型

1. 查看项目文件中的 `trained_model.pth`
2. 测试逻辑在 `core/training.py` 的 `test()` 函数中
3. 训练完成后，`main.py` 会自动运行测试

### 可视化结果

打开生成的 PNG 文件：
- `final_metrics.png` - 综合指标
- `battery_heatmap.png` - 电池热力图

在 PyCharm 中：右键文件 → "Open In" → "Image Viewer"

---

## 🎯 提示

1. **首次运行** 可能需要较长时间来安装依赖
2. **训练时间** 取决于 `max_steps` 参数，当前设置为 100,000 步
3. **GPU 加速**: 如果有 NVIDIA GPU，可以安装 CUDA 版本的 PyTorch
4. **调试模式**: 使用断点调试代码，按 `Ctrl + F8` 设置断点

---

## 📚 参考文档

- **PyCharm 官方文档**: https://www.jetbrains.com/pycharm/documentation/
- **Python 官方文档**: https://docs.python.org/3.10/
- **PyTorch 官方文档**: https://pytorch.org/docs/
- **Gym 文档**: https://gymnasium.farama.org/

---

**需要帮助？** 检查 `CLAUDE.md` 文件获取更多项目信息。