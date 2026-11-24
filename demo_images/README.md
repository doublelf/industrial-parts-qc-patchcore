# demo_images —— 演示图片说明

> ⚠️ 这些图片仅用于**界面演示**，模型是在 MVTec AD（CC BY-NC-SA 4.0）上建的记忆库，
> **不是真实产线对接**，界面里的不良率/精度不代表真实产线水平，禁止商用。

## 默认行为

不需要手动放图片：启动 `python app.py` 时，如果 `demo_images/<类别>/` 下没有图片，
服务会自动使用该类别 MVTec AD 的 **test 集**（`data/mvtec/<类别>/test`，good 与各缺陷类型混合），
并用固定 seed=42 打乱顺序，模拟良品/不良品随机到达。

## 换成自有演示图片

1. 按类别建目录（类别名必须与已训练类别一致）：

```
demo_images/
├── metal_nut/
│   ├── a_001.jpg
│   ├── a_002.png
│   └── ...
└── screw/
    └── ...
```

2. 支持 `.png/.jpg/.jpeg/.bmp/.tif`，可放子目录（递归扫描），图片尺寸任意（服务内部 Resize 到 256×256）。
3. **重启服务**生效（启动时一次性扫描并加载）。
4. 图片是 OK 还是 NG 不需要命名标注：界面显示的是模型打分和阈值判定；
   若你希望统计“不良率”接近真实比例，请按真实的良品/不良品比例放置图片。

## 从 MVTec test 集复制一批做固定演示

```bash
# 每个类别复制 12 张正常 + 每个缺陷类型 4 张到 demo_images/
python scripts/prepare_demo_images.py --category metal_nut
python scripts/prepare_demo_images.py --category screw
```

复制后 `demo_images/<category>/` 会包含 `good/` 与各缺陷类型子目录，重启服务即优先使用这些图片。
想恢复“默认使用完整 test 集”，删除对应 `demo_images/<category>/` 目录即可。

## 关于摄像头模式

摄像头抓拍的是**真实场景**，但模型没见过你的零件分布，结果仅用于演示交互效果，
不能用于判断真实零件是否合格。
