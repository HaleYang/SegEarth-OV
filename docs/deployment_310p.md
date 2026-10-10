# SegEarth-OV Ascend 310P 离线部署与推理评估指南

> PyTorch 权重 -> ONNX -> Ascend310P3 OM -> ACL Demo/UDD5 评测

## 1. 适用范围

本文档说明如何在 Ascend 310P3 上部署 SegEarth-OV，并复现单图 Demo、UDD5
全量精度评测和性能采样。现有 `docs/deployment.md` 主要记录 910/310B 流程，
310P 应使用本文档中的专用导出和编译脚本。

当前部署将模型拆成三个静态子模型：

| 子模型 | 功能 | 输入 | 输出 |
|---|---|---|---|
| `clip_visual` | CLIP 视觉编码器 | `image [1,3,224,224]` fp16 | `cls_token [1,512]`、`patch_tokens [1,196,512]` |
| `clip_text` | CLIP 文本编码器 | `text [8,77]` int64 | `text_features [8,512]` fp16 |
| `jbu_upsampler` | JBU 特征上采样 | `source [1,512,14,14]`、`guidance [1,3,224,224]` fp16 | `upsampled [1,512,224,224]` fp16 |

推理阶段的处理链路为：

```text
类别文本 -> clip_text.om -> 文本特征缓存
图像 -> 滑窗裁切 -> clip_visual.om -> jbu_upsampler.om
     -> 图文相似度 -> softmax/argmax -> 分割掩码
```

文本特征仅在缓存不存在时计算。正常的重复推理主要运行 Visual 和 JBU 两个 OM。

## 2. 已验证环境

| 项目 | 已验证版本 |
|---|---|
| 硬件 | Ascend 310P3，单卡约 21 GiB |
| SoC | `Ascend310P3` |
| CANN | 9.1.1 |
| Conda 环境 | `SegEarth310` |
| Python | 3.10.21 |
| PyTorch | 2.12.0+cpu |
| torch-npu | 2.12.0.post2 |
| torchvision | 0.27.0+cpu |
| ONNX | 1.23.0 |
| ONNXScript | 0.7.2 |
| mmcv | 2.1.0 |
| mmengine | 0.10.7 |
| mmsegmentation | 1.2.2 |
| NumPy | 2.2.6 |
| timm | 1.0.30 |

`torch`、`torch-npu` 和 `torchvision` 必须使用相互兼容的版本，不建议分别使用
无上限的 `>=`。CANN 不属于 pip 依赖，每个新终端都必须先加载其环境脚本。

文档中的占位符含义如下，执行前替换为实际安装位置：

| 占位符 | 含义 |
|---|---|
| `<CONDA_ROOT>` | Conda 安装目录 |
| `<CANN_ROOT>` | CANN 环境目录，目录下应有 `set_env.sh` |
| `<UDD_DATA_ROOT>` | 已解压的 UDD 数据集根目录 |

## 3. 准备模型权重和数据集

部署前需要准备两类权重和一份评测数据集：

1. Ascend 310P3 驱动、固件和 CANN 9.1.1 可用。
2. `SegEarth310` 环境已安装项目 Python 依赖和 `torch-npu`。
3. CLIP ViT-B/16 的 OpenAI 预训练权重可被 `create_model('ViT-B/16', pretrained='openai')` 加载。
   首次导出时，OpenCLIP 会按其缓存配置下载；也可以提前准备好对应的本地缓存。
4. JBU 权重已下载到：
   `simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt`。
5. 从 [UDD 官方仓库](https://github.com/MarcWong/UDD) 下载并解压 UDD 数据集。
   评测只需要 UDD5 验证集，但建议保留原始目录结构。

将数据集根目录链接到仓库：

```bash
mkdir -p data/UDD
ln -s <UDD_DATA_ROOT> data/UDD/UDD
```

确认目录最终可解析到 `data/UDD/UDD/UDD5/val/{src,gt}`：

```bash
find -L data/UDD/UDD/UDD5/val/src -type f | wc -l
find -L data/UDD/UDD/UDD5/val/gt -type f | wc -l
```

当前完整 UDD5 验证集应包含 40 张图和 40 张标注。

仓库内的 `open_clip/` 是本地源码，不需要额外安装 `open-clip-torch`。

## 4. 初始化环境和设备

进入仓库并加载环境：

```bash
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate SegEarth310
source <CANN_ROOT>/set_env.sh
cd SegEarth-OV
```

选择一张空闲物理卡。下面示例使用第二张物理卡：

```bash
export ASCEND_RT_VISIBLE_DEVICES=1
```

设置该变量后，进程内只看到一张卡，所以后续命令仍使用 `--device-id 0`。
如果改用其他物理卡，只修改 `ASCEND_RT_VISIBLE_DEVICES`，不要同时把
`--device-id` 改成物理编号。

部署前可检查环境：

```bash
npu-smi info
python -c "import torch, torch_npu, onnx; print(torch.__version__, torch_npu.__version__, onnx.__version__)"
```

整个导出、编译和推理流程均只需要一张 310P。

## 5. 导出 310P ONNX

310P 使用专用脚本 `scripts/export_onnx_310p.py`。Visual/Text 在 NPU 上导出，
JBU 由独立 CPU 进程导出，避免在同一进程中将活动 NPU 模块搬回 CPU 引发流或
内存分配错误。

```bash
export SEG310P_OUTPUT_DIR=models/onnx_310p
export SEG310P_JBU_UNFOLD_MODE=slice
python scripts/export_onnx_310p.py
```

产物如下：

```text
models/onnx_310p/
├── clip_visual.onnx
├── clip_visual.onnx.data
├── clip_text.onnx
├── clip_text.onnx.data
├── jbu_upsampler.onnx
└── jbu_upsampler.onnx.data
```

### 5.1 310P 图适配

310P 导出保留显式 Attention，并应用以下处理：

| 模块/算子 | 310P 导出策略 | 原因 |
|---|---|---|
| Attention | 显式 Q/K/V、MatMul、Softmax 图 | 当前 torch-npu MHA 路径在导出前可能触发格式转换错误 |
| Text `argmax` | 恢复原生 `argmax` | 310P/CANN 可编译该图，避免沿用 310B 兼容替代 |
| JBU patch 提取 | 静态 Slice 路径 | 避免 Unfold 导出的 GatherV2 成为性能瓶颈 |
| Bicubic 2x | depthwise conv 等价实现 | CANN 9.1.1 不接受 `Resize(bicubic, half_pixel)` |
| Attention Q/K/V 拆分 | 显式 Slice | 避免 CANN 拒绝三输出 Split |

JBU Slice ONNX 节点较多，CPU 导出可能需要十几分钟。进程持续占用 CPU 时应继续
等待，不要用旧 JBU 文件替换本次产物。

### 5.2 Text batch 固定为 8

当前 Text 图是静态 `[8,77]`：

- 导出脚本使用 8 条 dummy text，并关闭动态轴；
- ATC 编译显式指定 `text:8,77`；
- ACL 运行时每 8 条模板分一批，最后一批用零 token 补齐，推理后截掉补齐结果。

因此，修改 Text batch 时必须同步修改导出、ATC 输入 shape 和
`utils/session.py` 的分批/补齐逻辑，不能只改其中一处。

### 5.3 ONNX opset 说明

PyTorch 2.12 即使请求 opset 17，也可能输出 opset 18，并显示自动降级失败的警告。
这不是当前流程的导出失败。`scripts/build_om_310p.sh` 会复制 ONNX 到临时目录，
只调整临时副本的标准域 opset 元数据，再交给 ATC；原始 ONNX 不会被修改。

可检查 ONNX 是否能正常加载：

```bash
python - <<'PY'
from pathlib import Path
import onnx

for path in sorted(Path("models/onnx_310p").glob("*.onnx")):
    model = onnx.load(str(path), load_external_data=False)
    versions = [(item.domain, item.version) for item in model.opset_import]
    print(path, versions, len(model.graph.node))
PY
```

## 6. 编译 Ascend310P3 OM

```bash
export ONNX_DIR=models/onnx_310p
export OM_DIR=models/om_310p
export SOC_VERSION=Ascend310P3
bash scripts/build_om_310p.sh
```

三个模型的静态输入为：

| 模型 | ATC `input_shape` |
|---|---|
| `clip_visual` | `image:1,3,224,224` |
| `clip_text` | `text:8,77` |
| `jbu_upsampler` | `source:1,512,14,14;guidance:1,3,224,224` |

编译结果：

```text
models/om_310p/
├── clip_visual.om
├── clip_text.om
└── jbu_upsampler.om
```

编译日志位于 `logs/om_310p/`。Text 编译时可能提示 `ArgMaxV2` 未命中
高优先级算子库；只要 ATC 最终输出 `ATC run success`，该提示不是编译失败。

检查产物：

```bash
ls -lh models/om_310p/*.om
```

## 7. 单图 Demo 推理

下面使用仓库示例遥感图和 9 类配置：

```bash
python scripts/demo_om.py \
  -i demo/oem_koeln_50.tif \
  -o demo_output_om_310p \
  --config configs/cfg_demo9_om.py \
  --om-dir models/om_310p \
  --device-id 0 \
  --size 448 \
  --template full
```

输出包括：

```text
demo_output_om_310p/
├── oem_koeln_50_mask.png
├── oem_koeln_50_mask_color.png
└── oem_koeln_50_vis.jpg
```

其中 `mask.png` 保存类别索引，适合程序处理；`mask_color.png` 和 `vis.jpg` 适合
人工查看。首次运行会调用 Text OM 并在 `models/om_310p/` 生成
`query_features_*.npy`。类别和模板不变时，后续运行直接读取缓存。

需要同时记录推理和整条命令耗时时，可执行：

```bash
mkdir -p logs/demo_om_310p
/usr/bin/time -f "WALL_TIME=%e\nUSER_TIME=%U\nSYSTEM_TIME=%S\nCPU_PERCENT=%P\nMAX_RSS_KB=%M" \
  python -u scripts/demo_om.py \
    -i demo/oem_koeln_50.tif \
    -o demo_output_om_310p \
    --config configs/cfg_demo9_om.py \
    --om-dir models/om_310p \
    --device-id 0 \
    --size 448 \
    --template full \
  2>&1 | tee logs/demo_om_310p/run.log
```

- `infer`：只包含该图片的模型预测时间；
- `vis`：生成可视化结果的时间；
- `WALL_TIME`：从 Python 启动到进程退出的实际等待时间，包含导包、ACL 初始化、
  OM 加载、推理和结果保存。

当前最新 OM 的 9 类 Demo 实测：

| 场景 | 推理 | 可视化 | 命令墙钟时间 |
|---|---:|---:|---:|
| 首次运行，生成文本特征缓存 | 8.62 s | 0.11 s | 27.76 s |
| 缓存命中 | 8.63 s | 0.11 s | 26.09 s |

该表只适用于 `demo/oem_koeln_50.tif`、9 类、`size=448`、`template=full` 的当前
测试口径。

## 8. UDD5 精度和性能评测

### 8.1 快速冒烟测试

先用一张图确认数据、模型和 ACL Runtime 均可用：

```bash
python scripts/eval_acl.py \
  --config configs/cfg_udd5.py \
  --backend acl \
  --om-dir models/om_310p \
  --device-id 0 \
  --size 448 \
  --max-samples 1 \
  --template full \
  --debug-timing
```

### 8.2 完整 40 张评测

```bash
mkdir -p logs/eval_udd5_310p
python -u scripts/eval_acl.py \
  --config configs/cfg_udd5.py \
  --backend acl \
  --om-dir models/om_310p \
  --device-id 0 \
  --size 448 \
  --max-samples 0 \
  --template full \
  --debug-timing \
  2>&1 | tee logs/eval_udd5_310p/eval.log
```

`--max-samples 0` 表示使用全部样本。`--debug-timing` 会输出每个 crop 的 Visual、
JBU 和 NumPy 后处理时间；精度计算本身不依赖该选项。

### 8.3 当前实测结果

最新重新生成的 Slice JBU OM 在 UDD5 验证集 40 张上的结果：

| 指标 | 结果 |
|---|---:|
| aAcc | 74.82% |
| mIoU | 50.51% |
| mAcc | 66.10% |
| 总推理时间 | 265.27 s |
| 平均推理时间 | 6.632 s/image |
| 吞吐量 | 0.15 image/s |

逐类结果：

| 类别 | IoU | Acc |
|---|---:|---:|
| vegetation | 80.54% | 85.95% |
| building | 70.87% | 77.61% |
| road | 53.27% | 83.20% |
| vehicle | 21.52% | 39.85% |
| background | 26.37% | 43.89% |

该结果与项目历史 ACL/OM 的 `50.56%` mIoU 相差 0.05 个百分点，可视为精度
保持稳定。历史数据的硬件和性能统计口径与当前 310P 不完全一致，不应据此直接
计算跨硬件加速比。

## 9. PyTorch 基线对比

使用相同 UDD5 配置运行 310P PyTorch 基线时，需要显式关闭原 310B 兼容 patch：

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
python -u -c 'import runpy,sys; from npu_compat import setup_npu; setup_npu(use_patch=False,use_interpolate_patch=False); sys.argv=["scripts/eval_acl.py","--config","configs/cfg_udd5.py","--backend","pytorch","--device-id","0","--size","448","--max-samples","0","--template","full"]; runpy.run_path("scripts/eval_acl.py",run_name="__main__")'
```

同一 UDD5 全量口径下的已有对比：

| 后端 | aAcc | mIoU | mAcc | 平均推理 | 吞吐量 |
|---|---:|---:|---:|---:|---:|
| PyTorch patch-off | 74.62% | 50.38% | 65.89% | 14.226 s/image | 0.07 image/s |
| 310P Slice OM | 74.82% | 50.51% | 66.10% | 6.632 s/image | 0.15 image/s |

当前口径下 OM 平均推理约为 PyTorch 的 2.15 倍速度。两条路径的 JBU 和数值实现
不同，指标应比较整体精度，不要求逐像素完全一致。

## 10. 资源占用采样

评测运行期间，可在另一个终端检查 NPU：

```bash
npu-smi info
```

查找评测进程并采样 CPU/内存：

```bash
pgrep -af "scripts/eval_acl.py"
ps -p <PID> -o pid,pcpu,pmem,rss,cmd
```

如果系统安装了 `pidstat`，可获得更稳定的短间隔采样：

```bash
pidstat -p <PID> 1 10
```

当前 UDD5 ACL 全量评测的观测值：

| 资源 | 实测范围 |
|---|---:|
| NPU 总显存 | 8379-8429 / 21524 MB |
| NPU 显存占比 | 38.9%-39.2% |
| Python 进程 NPU 显存 | 约 6581-6629 MB |
| 评测进程 CPU | 稳定约 70%-71%，阶段性约 90% |
| 评测进程 RSS | 约 2.1-3.1 GiB |
| AICore 利用率采样 | 约 45%-98% |

Linux `ps` 的 CPU 百分比以单个逻辑核为 100%。在 64 核服务器上，进程显示
`70%` 不等于使用整机 70%，而是约 0.7 个逻辑核。

## 11. 常见问题

### 11.1 `ModuleNotFoundError` 或 ACL 库找不到

确认先激活 `SegEarth310`，再执行 `source <CANN_ROOT>/set_env.sh`。每个新终端都
需要重新执行。

### 11.2 设备号与物理卡不一致

设置 `ASCEND_RT_VISIBLE_DEVICES=1` 后，第二张物理卡在进程中映射为设备 0，
因此命令使用 `--device-id 0`。

### 11.3 Text OM 输入 shape 错误

当前 `clip_text.om` 固定输入 `[8,77]`。运行时必须按 8 条切分并补齐；不要把任意
batch 的 token 直接送入该 OM。

### 11.4 ONNX 显示 opset 18

这是 PyTorch 2.12 导出器的已知行为。使用 `scripts/build_om_310p.sh` 编译即可，
不要手工覆盖原始 ONNX 外部权重文件。

### 11.5 JBU 导出或编译耗时较长

Slice JBU 图包含大量静态 Slice 节点，CPU 导出和 ATC 编译都明显慢于 Visual/Text。
只要进程仍持续占用 CPU 且没有异常退出，应等待完成。

### 11.6 输出掩码全黑或类别异常

依次确认：

1. `--config` 指向正确类别文件；
2. `--template` 与文本特征缓存口径一致；
3. 使用当前 `models/om_310p` 的三个配套 OM；
4. 删除不再适用的 `query_features_*.npy` 后重新生成文本特征；
5. 查看原始索引掩码的唯一值，而不是只凭默认调色板判断。

## 12. 完整复现顺序

```text
1. 激活 SegEarth310，并加载 CANN 环境
2. 选择一张空闲 310P
3. 检查权重和 UDD5 数据路径
4. export_onnx_310p.py 生成三个 ONNX
5. build_om_310p.sh 生成三个 OM
6. demo_om.py 完成单图冒烟验证
7. eval_acl.py --max-samples 1 完成数据集冒烟验证
8. eval_acl.py --max-samples 0 完成 UDD5 全量精度/性能评测
9. 评测期间用 npu-smi 和 ps/pidstat 采样资源占用
```

提交代码时通常只提交脚本、配置和文档。`models/onnx_310p/`、
`models/om_310p/`、推理输出和评测日志均属于生成产物，不建议直接提交到 Git。
