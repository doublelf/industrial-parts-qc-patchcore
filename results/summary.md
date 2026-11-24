# PatchCore 多类别汇总对比

- 数据集：MVTec AD（CC BY-NC-SA 4.0，仅研发验证）
- 随机种子：42
- backbone：wide_resnet50_2 | layers：['layer2', 'layer3'] | coreset 比例：0.1 | 输入：256x256
- 阈值策略：val_normal_robust（验证集正常样本 median + k×1.4826×MAD，k=3.0；不使用 test）

## 1. 指标与耗时对比

| 类别 | 图像 AUROC | 图像 AP | 像素 AUROC | 像素 AP | AU-PRO | 阈值 | FP | FN | 单张 mean | 单张 p99 | 设备 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| metal_nut | 0.9980 | 0.9995 | 0.9870 | 0.8864 | 0.9410 | 40.6582 | 0 | 2 | 607.3 ms | 757.2 ms | CPU: 16 threads |
| screw | 0.9520 | 0.9827 | 0.9895 | 0.3369 | 0.9411 | 34.1558 | 4 | 25 | 429.7 ms | 533.7 ms | CPU: 16 threads |

## 2. 典型失败案例与边界样本（每类 3 个）

| 类别 | 情况 | 文件 | 分数 | 说明 |
|---|---|---|---:|---|
| metal_nut | 漏检 | 004.png | 39.2988 | color 缺陷分数低于阈值 |
| metal_nut | 漏检 | 007.png | 40.3139 | bent 缺陷分数低于阈值 |
| metal_nut | 漏检 | 000.png | 40.9675 | scratch 缺陷分数低于阈值 |
| metal_nut | 正常最高分 | 016.png | 40.5111 | 正常样本中分数最高，未超过阈值 |
| metal_nut | 正常最高分 | 007.png | 39.8487 | 正常样本中分数最高，未超过阈值 |
| metal_nut | 正常最高分 | 011.png | 39.7954 | 正常样本中分数最高，未超过阈值 |
| screw | 漏检 | 022.png | 29.7286 | scratch_head 缺陷分数低于阈值 |
| screw | 漏检 | 013.png | 29.8849 | thread_side 缺陷分数低于阈值 |
| screw | 漏检 | 003.png | 30.0120 | thread_top 缺陷分数低于阈值 |
| screw | 正常最高分 | 035.png | 35.3310 | 正常样本中分数最高，未超过阈值 |
| screw | 正常最高分 | 005.png | 35.2452 | 正常样本中分数最高，未超过阈值 |
| screw | 正常最高分 | 025.png | 34.5842 | 正常样本中分数最高，未超过阈值 |

## 3. 复现信息

| 类别 | seed | Anomalib | 测试集数量 | 阈值来源样本数 |
|---|---:|---|---:|---:|
| metal_nut | 42 | 2.7.1 | 115 | 44 |
| screw | 42 | 2.7.1 | 160 | 64 |

> 所有指标均来自 `results/<category>/metrics.json` 的实际运行输出；
> 耗时优先取自 `results/<category>/latency_bench.json`（20 张 × 5 次端到端，含读图），
> 没有基准文件时回退到 `evaluate.py` 的测量值。CPU 共享开发机耗时存在 ±30% 波动。
