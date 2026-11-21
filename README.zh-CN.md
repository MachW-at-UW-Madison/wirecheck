[English](README.md) | 简体中文

# WireCheck 复向量分析工具

从 iPad VIO 位姿与三轴磁场录音出发，估计电缆位置、负载电流与漏电流，并给出基于固定阈值的健康判定。

> **如果你在工作中使用了 WireCheck、其代码、其数据或四点扫描，请引用我们的论文：**
>
> Xincheng Xie and Akarsh Prabhakara. 2026. Demo: Locating Leakage Current via Magnetometers. In *The 32nd Annual International Conference on Mobile Computing and Networking (MobiCom '26)*. ACM. https://doi.org/10.1145/3795866.3848441
>
> BibTeX 见下文[引用](#引用)一节；GitHub 的 “Cite this repository” 按钮读取 [CITATION.cff](CITATION.cff)。

> **0.2 新增：四点扫描。** 在电缆两侧低处各取一近一远两个点，组成两对镜像点，每点短暂停留，约 40 秒即可得到漏电估计。原理见 [FOUR-POINT.zh-CN.md](FOUR-POINT.zh-CN.md)。为此，现在每个块都以同一个固定工频、按绝对时间解调，整次采集共用一个锁相相位。

一次采集（campaign）对应一个目录，目录里有两个 CSV：

- `raw.csv`：三轴磁场录音；
- `pose.csv`：iPad（ARKit）视觉惯性里程计（VIO）位姿，约 60 Hz。

`WireCheck` 把连续磁场录音按固定时长切成锁相块，用每块中点时刻的 VIO 位姿作为该测量点的位置，并把该块的传感器样本旋转到统一坐标系；随后对每块提取工频（50/60 Hz）复数相量，并拟合一个物理模型：

- 双导线对（火线 + 零线，反平行线电流）产生的精确近场；
- 单极漏电流项 `C / r`（以同相/正交的笛卡尔复分量表示）；
- 均匀的相干背景场 `U`（一个复三维矢量，作为干扰项）。

默认情况下，每个块都以同一个固定工频解调，相位以磁力计的绝对时间为参考，所以所有块共用一个锁相相位，拟合对整次采集只求一次相位（见[锁相相位](#锁相相位)）。

拟合输出电缆中心位置、对偶电流、漏电流幅值与相对相位，并据此给出判定：`no_cable`、`healthy`、`leakage_suspected` 或 `leakage_detected`。

## 环境要求

- Python >= 3.10
- 运行时依赖：`numpy >= 1.24`、`scipy >= 1.11`、`matplotlib >= 3.7`、`tqdm >= 4.66`
- 可选加速：`numba >= 0.60`。仅用于加速残差与雅可比内核；未安装时会回退到等价的 NumPy 实现，结果不受影响。

## 安装

在本仓库根目录执行：

```bash
# 正式安装
pip install .

# 开发模式（可编辑安装）
pip install -e .

# 同时安装 numba 加速
pip install ".[speed]"
```

安装后提供命令行程序 `wirecheck`：

```bash
wirecheck --help
```

### 从源码构建 wheel

```bash
python -m pip install build
python -m build
python -m pip install dist/magcomm_wirecheck-0.2.0-py3-none-any.whl
```

## 数据格式

### 1. `raw.csv`：磁场采集

逗号分隔，第一行为表头。必须包含以下列（列顺序可任意，`sample` 列可以没有）：

| 列 | 含义 |
| --- | --- |
| `t_host` | 采样时刻，单位秒（采集开始以来的秒数，与 `pose.csv` 同一时钟） |
| `t_mcu` | 微控制器时刻，单位秒，必须严格递增 |
| `x` | X 轴原始 ADC 计数 |
| `y` | Y 轴原始 ADC 计数 |
| `z` | Z 轴原始 ADC 计数 |

- ADC 为 18 位，量程中点为 `2^17 = 131072`。原始计数按以下公式换算为纳特斯拉：

  ```text
  B[nT] = (raw - 131072) * (100000 / 16384)   # 约 6.1035 nT / LSB
  ```

- 启动裁剪：`t_mcu <= 0.1 s` 的样本会被丢弃。
- 采样率由 `1 / median(diff(t_mcu))` 估计，用于确定每块的样本数。

示例：

```csv
sample,t_host,t_mcu,x,y,z
0,0.200000,0.200000,131080,131071,131090
1,0.200800,0.200800,131082,131069,131088
```

### 2. `pose.csv`：iPad VIO 位姿

逗号分隔，第一行为表头，必须包含：

| 列 | 含义 |
| --- | --- |
| `t_host` | 位姿时刻，单位秒；与 `raw.csv` 同一时钟，必须严格递增 |
| `x` | 相机位置 X，单位米 |
| `y` | 相机位置 Y，单位米（ARKit 世界坐标，+Y 向上） |
| `z` | 相机位置 Z，单位米 |
| `qw` `qx` `qy` `qz` | 相机朝向四元数（ARKit 约定） |

程序会自动处理相邻帧之间的四元数符号翻转，并按 `t_host` 线性插值到每个块的中点。

示例：

```csv
t_host,x,y,z,qw,qx,qy,qz
0.200000,0.000000,0.040000,0.100000,1,0,0,0
0.216667,0.000000,0.040000,0.100000,1,0,0,0
```

### 3. `meta.json`（可选）

如果采集目录里有 `meta.json` 且其中有 `{"pi": {"line_hz": F}}`，`F` 就作为锁相工频（见下文）。不读取其他字段。

### 4. 采集目录

`--data` 必须是一个目录，目录里必须同时存在 `raw.csv` 与 `pose.csv`。一个目录就是一个 collection。

## 位置与坐标系

位置**不再来自文件名**，而是来自 `pose.csv` 的时间插值：

- 录音按固定时长（`--block-seconds`，默认 0.5 s）切成不重叠的锁相块；每个块成为一个测量点。
- 每个块取**中点时刻**的相机位姿作为位置。
- 分析平面由**第一个有效位姿**定义：
  - 水平轴：首帧相机前向在水平面（世界 XZ 平面）上的投影；
  - 竖直轴：世界 +Y；
  - 沿电缆轴：两者叉积。
- 每个块的位置是该块中点时刻的传感器位置在分析平面上的投影（水平、竖直两个分量）；沿电缆方向的分量用于门限判定，不进入拟合。
- 每个块的三轴样本会用该中点时刻的相机姿态旋转到统一坐标系；这样即使操作者转动了 iPad，各块的相量仍在同一坐标系中。
- 杆臂与传感器朝向通过 `--lever-mm` 与 `--sensor-rotation` 给出；默认分别为 `0,0,0` 与单位矩阵。

为避免破坏"单个测量点位置固定"的假设，两个门限会丢弃不合格的块：

- `--max-travel-mm`（默认 50）：块内相机移动超过该值则丢弃；
- `--max-along-mm`（默认 200）：该块沿电缆方向的偏移超过该值则丢弃，使扫描保持在一个横截面内。

## 锁相相位

每个块的相量为

```text
B_k = 2 / (sqrt(2) * sum(w)) * sum_n w_n (x(t_n) - mean_k x) exp(-i 2 pi f_L t_n)
```

其中 `w` 为 Hann 窗，`t_n` 为绝对的 `t_mcu` 时间，因此每个块带有同一个未知相位 `g`。拟合对整次采集只求这一个相位，而不是每块一个：

```text
g = C / |C|,   C = sum_k p_k* . B_k / sigma_k^2,   r_k = (B_k - g p_k) / sigma_k
```

其中 `p_k` 是块 `k` 处的模型磁场。工频 `f_L`（磁力计时钟下）按以下顺序取得：

| 来源 | 何时使用 |
| --- | --- |
| `--line-hz F` | 命令行给出 |
| `meta.json` 的 `pi.line_hz` | 采集程序记录了该值 |
| 从录音估计 | 其余情况：对偶项、漏电项与背景场同相，所以 `arg(sum_j B_kj^2)` 以 `4 pi df t` 前进，其在强信号块上的斜率给出 `df` |

如果估计时强信号块少于五个，或者给了 `--per-block-phase`，则像 0.1 一样每块各自追踪频率与相位。共用一个相位要求一次扫描内磁力计时钟下的工频恒定（信号发生器或稳定电源）。1 mHz 的误差在 85 秒的扫描中会使相位转约 30 度，所以对频率漂移的电源（例如真实市电）请使用 `--per-block-phase`。

## 运行

命令行提供两个子命令，二者共享同一组 campaign 参数。

### analyze：分析单个 campaign

```bash
wirecheck analyze --data path/to/campaign
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--block-seconds` | 0.5 | 锁相块长度，单位秒；一个块 = 一个测量点 |
| `--lever-mm X Y Z` | `0 0 0` | 相机到传感器的杆臂，单位 mm，在相机坐标系中 |
| `--sensor-rotation R00,...,R22` | 单位矩阵 | 传感器坐标轴到相机坐标轴的 3×3 旋转（行优先，逗号分隔） |
| `--max-travel-mm` | 50 | 块内相机位移超过该值则丢弃 |
| `--max-along-mm` | 200 | 沿电缆方向偏移超过该值则丢弃 |
| `--line-hz` | 见[锁相相位](#锁相相位) | 共用相位锁相所用的工频 |
| `--per-block-phase` | 关闭 | 每块各自追踪频率与相位，而不是整次一个相位 |
| `--z1-height-mm` | 10000 | 拟合电缆深度的物理下界（取负值作为下界，单位 mm） |
| `--core-mm` | 0 | 导体半径，单位 mm；0 表示使用未平滑的精确双导线场 |
| `--pair-spacing-mm` | 5.5 | 火线/零线中心间距，单位 mm；同时决定近场形状与电流换算 |
| `--fractional-floor` | 0.094 | 固定的模型误差比例（与统计误差按平方和合成） |
| `--no-figure` | 关 | 不生成诊断图 |
| `--show` | 关 | 交互式显示诊断图 |
| `--out-dir DIR` | 无 | 指定输出目录 |

输出：

- 标准输出：文本报告；
- `<data>/vector-lockin-output/result.json`：完整结果，含所用的 `line_hz`（每块各自相位时为 `null`）；
- `<data>/vector-lockin-output/figures/<collection>_summary.png`：诊断图（除非指定 `--no-figure`）。

### calibrate-floor：标定 fractional floor

使用两个或更多"已知健康"、且扫描计划与坐标系完全一致的 campaign，估计一个共享的固定 `fractional_floor`：

```bash
wirecheck calibrate-floor --data healthy-1 healthy-2 [healthy-3 ...]
```

可选参数：`--seed-floor`、`--search-min`、`--search-max`、`--output FILE`。结果既打印到标准输出，也可用 `--output` 写入 JSON 文件。

所有参与标定的 campaign 必须使用相同坐标系与相同测点集合，各自独立拟合，仅共享 `fractional_floor` 一个参数。

## 判定规则

判定使用固定阈值：

| 条件 | 判定 |
| --- | --- |
| 拟合负载电流 < 0.1 A | `no_cable` |
| 漏电流 < 12 mA | `healthy` |
| 12 mA ≤ 漏电流 < 18 mA | `leakage_suspected` |
| 漏电流 ≥ 18 mA | `leakage_detected` |

其中漏电流由漏电系数 `C`（单位 nT·m）换算：`I_leak[mA] = 5 × |C|`；对偶电流 `I_pair[A] = 5 × a / pair_spacing[mm]`，`a` 为拟合的对偶系数（单位 nT·m²）。负载电流为 `I_pair + I_leak / 2000`。

## 不安装直接运行

本包要求导入名为 `wirecheck`，而本仓库目录本身就是该包。因此不安装时，需要让一个名为 `wirecheck` 的目录出现在 `PYTHONPATH` 上。

方式一：在任意位置创建符号链接。

```bash
# 在仓库根目录执行
mkdir -p /tmp/wc && ln -sfn "$PWD" /tmp/wc/wirecheck
PYTHONPATH=/tmp/wc python -m wirecheck.wirecheck_cli analyze --data path/to/campaign
```

方式二：把仓库目录重命名为 `wirecheck`，然后在其父目录执行：

```bash
python -m wirecheck.wirecheck_cli --help
```

两种方式都无需安装，也不需要修改源码。

## Python API

```python
from wirecheck import file_adapter, vector

collection = file_adapter.load_campaign(
    "path/to/campaign",
    block_seconds=0.5,
    lever_arm_m=(0.0, 0.0, 0.0),
)
analysis = vector.analyze(collection, vector.Config(block_seconds=0.5))

print(analysis.report["health"])
print(analysis.report["estimated_leak_current_ma"])
figure = analysis.make_figure()
```

需要把传感器样本旋转到相机坐标系时，给 `load_campaign` 传入 `sensor_to_camera=` 一个 3×3 矩阵。传入 `line_hz=F` 可固定锁相工频，传入 `per_block_phase=True` 则每块一个相位；`collection.reference_hz` 为实际使用的工频（每块各自相位时为 `None`）。

## 目录结构

```text
.
├── pyproject.toml
├── LICENSE                 # GPL-3.0-or-later
├── CITATION.cff            # 引用元数据
├── FOUR-POINT.md           # 四点扫描为何有效
├── docs/                   # FOUR-POINT.md 的图
├── wirecheck_cli.py        # 命令行入口
├── file_adapter.py         # raw.csv / pose.csv 读取、时钟与坐标变换
└── vector/                 # 与采集格式无关的分析实现
    ├── defs.py             # 结构化协议
    ├── rdata.py            # 原始记录与采集契约
    ├── preproc.py          # 锁相频谱提取（多块平均、单块、固定参考三种入口）与工频估计
    ├── meas.py             # 归一化测量与集合
    ├── core.py             # 系数拟合与手性选择
    ├── _vector_kernel.py   # 可选的 numba 残差/雅可比内核
    ├── calibrate.py        # fractional floor 标定
    ├── parse.py            # 硬阈值、质量门与判定
    ├── plot.py             # 绘图
    └── wirecheck.py        # 组合各模块的服务接口
```

## 假设与限制

- 电缆大致水平，扫描在一个近似垂直于电缆的平面内进行。
- iPad VIO 提供相机位姿；传感器相对相机的旋转与杆臂需要标定（`--sensor-rotation`、`--lever-mm`），默认是单位旋转与零杆臂。
- `raw.csv` 与 `pose.csv` 使用同一时钟 `t_host`。
- 每个锁相块内近似静止；位移超过 `--max-travel-mm` 的块会被丢弃。
- 共用相位的锁相假设一次扫描内磁力计时钟下的工频恒定；否则请使用 `--per-block-phase`。
- 保留各轴之间的 I/Q 相位关系；忽略轴间延迟。
- 相干背景场均匀且相位稳定。
- `pair_spacing` 同时决定近场形状与电流换算，必须与实际几何匹配。
- 质量门限基于有限数据调校；几何或安装变化时应重新标定 `fractional_floor`。

## 引用

任何使用本软件、其结果或四点扫描的论文、报告或产品，请引用我们的 demo 论文：

> Xincheng Xie and Akarsh Prabhakara. 2026. Demo: Locating Leakage Current via Magnetometers. In *The 32nd Annual International Conference on Mobile Computing and Networking (MobiCom '26), October 26–30, 2026, Austin, TX, USA.* ACM, New York, NY, USA. https://doi.org/10.1145/3795866.3848441

```bibtex
@inproceedings{xie2026wirecheck,
  author    = {Xie, Xincheng and Prabhakara, Akarsh},
  title     = {Demo: Locating Leakage Current via Magnetometers},
  booktitle = {The 32nd Annual International Conference on Mobile Computing and Networking (MobiCom '26)},
  year      = {2026},
  month     = oct,
  address   = {Austin, TX, USA},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3795866.3848441},
  isbn      = {979-8-4007-2505-0}
}
```

GitHub 的 “Cite this repository” 按钮从 [CITATION.cff](CITATION.cff) 读取同一条目。

## 许可证

本项目以 **GNU General Public License v3.0 或更高版本（GPL-3.0-or-later）** 发布，完整条款见 [LICENSE](LICENSE)。
