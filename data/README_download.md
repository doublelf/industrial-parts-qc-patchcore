# MVTec AD 数据下载说明

> **许可提醒**：MVTec AD 采用 **CC BY-NC-SA 4.0** 许可，**仅限非商业研发与学术验证**，
> 禁止直接用于商业产品/商业服务。引用请注明原作者：
> Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger,
> *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.

## 1. 官方渠道（推荐，需填表）

1. 打开官网：<https://www.mvtec.com/company/research/datasets/mvtec-ad>
2. 填写下载表格（姓名/邮箱/机构/用途），提交后获得下载链接。
3. 下载压缩包 `mvtec_anomaly_detection.tar.xz`（约 4.9 GB）。
4. 解压到本项目 `data/mvtec/`，确保解压后**直接**出现类别目录：

```
data/mvtec/
├── metal_nut/
│   ├── train/good/            # 220 张正常图
│   ├── test/good/             # 22 张正常图（测试）
│   ├── test/bent/*.png        # 25 张
│   ├── test/color/*.png       # 22 张
│   ├── test/flip/*.png        # 23 张
│   ├── test/scratch/*.png     # 23 张
│   └── ground_truth/{bent,color,flip,scratch}/*_mask.png
└── screw/
    ├── train/good/            # 320 张
    ├── test/good/             # 41 张
    ├── test/manipulated_front/*.png  # 24 张
    ├── test/scratch_head/*.png       # 24 张
    ├── test/scratch_neck/*.png       # 25 张
    ├── test/thread_side/*.png        # 23 张
    ├── test/thread_top/*.png         # 23 张
    └── ground_truth/<同上缺陷类型>/*_mask.png
```

如果解压出来是 `data/mvtec/mvtec_anomaly_detection/metal_nut/...`，
请把内层目录内容整体上移一层，或把 `config.yaml` 里的 `data.root` 指到内层目录。

## 2. 免填表的公开镜像（仅研发用途，同样受 CC BY-NC-SA 4.0 约束）

Hugging Face 上有保留官方目录结构的镜像（非官方分发，使用即表示接受原许可条款）：

```bash
# 只下载本项目需要的两个类别，避免拉取 5 GB 全量数据
huggingface-cli download foersben/mvtec-ad --repo-type dataset \
    --include "metal_nut/*" "screw/*" --local-dir data/mvtec
```

或直接使用项目自带脚本（内部调用 `huggingface_hub.snapshot_download`）：

```bash
python scripts/download_data.py --categories metal_nut screw
```

## 3. 校验

```bash
python scripts/check_data.py --all
```

预期输出（metal_nut / screw）：

| 集合 | 类型 | metal_nut | screw |
|---|---|---:|---:|
| train | good | 220 | 320 |
| test | good | 22 | 41 |
| test | bent | 25 | - |
| test | color | 22 | - |
| test | flip | 23 | - |
| test | scratch | 23 | - |
| test | manipulated_front | - | 24 |
| test | scratch_head | - | 24 |
| test | scratch_neck | - | 25 |
| test | thread_side | - | 23 |
| test | thread_top | - | 23 |

任何一项为 0，`check_data.py` 会打印告警并以非 0 退出码结束。

## 4. 常见问题

- **Windows 解压 `.tar.xz`**：WinRAR/7-Zip 均可；或用 `tar -xf mvtec_anomaly_detection.tar.xz -C data/mvtec`（Win10+ 自带 tar 支持 xz）。
- **类别名大小写**：必须是 `metal_nut`、`screw`，不要写成 `Metal_Nut`。
- **数据缺失时的行为**：`train.py` / `evaluate.py` 会打印与本文档一致的提示并以非 0 退出，不会抛裸异常。
