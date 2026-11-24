# ONNX 导出报告 — screw

- 状态：**success**
- 导出文件：`D:\Desktop\chat\opencode\project\industrial-parts-qc-patchcore\results\screw\export\weights\onnx\model.onnx`（合计 248.67 MB，含外部权重文件 .data）

## 数值验证（CPU，单张图）

- PyTorch 参考分数（raw PatchCore distance (no post-processor)）：30.332529
- 验证通过的输入格式：`resized_256_raw_[0,1]`

| 输入格式 | ONNX 分数 | 与 PyTorch 差值 | 异常图最大绝对误差 |
|---|---:|---:|---:|
| resized_256_raw_[0,1] | 30.332449 | 8.011e-05 | 0.000110626220703125 |
| resized_256_imagenet_normalized | 226.085037 | 1.958e+02 | 199.261962890625 |

推理验证建议（ONNX Runtime，CPU）：

```python
import numpy as np, onnxruntime as ort
from PIL import Image
sess = ort.InferenceSession('model.onnx', providers=['CPUExecutionProvider'])
img = Image.open('part.png').convert('RGB').resize((256, 256), Image.BILINEAR)
x = (np.asarray(img, np.float32) / 255.0).transpose(2, 0, 1)[None]  # (1,3,256,256)
score, anomaly_map = sess.run(None, {sess.get_inputs()[0].name: x})
# score 为原始 PatchCore 距离，与 results/<category>/threshold.json 中的阈值比较得到 OK/NG
```

## 步骤日志

- [x] 加载模型 D:\Desktop\chat\opencode\project\industrial-parts-qc-patchcore\results\screw\model.pt
- [x] ONNX 导出 D:\Desktop\chat\opencode\project\industrial-parts-qc-patchcore\results\screw\export\weights\onnx\model.onnx
- [x] ONNX Runtime 加载 input=input ['batch_size', 3, 256, 256] outputs=['pred_score', 'anomaly_map']
- [x] 数值验证 最大分数误差 8.01e-05
