# SegEarth-OV 310P 优化与指标汇总

> 记录日期：2026-09-29  
> 项目路径：`SegEarth-OV`  
> 目标硬件：Ascend 310P3  
> 运行环境：Conda `SegEarth310`，Python 3.10，CANN 环境脚本：`<CANN_ROOT>/set_env.sh`

本文档汇总当前项目从 PyTorch 到 310P OM 的适配、优化、验证结果，并区分本次 310P 同输入实测数据与原文档中的历史 310B/910 数据。

## 1. 推理结构

模型由三个子模型组成：

| 子模型 | 作用 | 输入 | 输出 |
|---|---|---|---|
| `clip_visual` | CLIP ViT-B/16 视觉编码 | `[1,3,224,224]` FP16 | cls token、patch token |
| `clip_text` | 文本类别特征编码 | `[8,77]` | 文本特征 `[8,512]` |
| `jbu_upsampler` | JBU 特征上采样 | `[1,512,14,14]` + `[1,3,224,224]` | `[1,512,224,224]` |

单图推理使用滑窗：输入尺寸 `448x448`，crop `224`，stride `112`。视觉 OM 得到 patch feature，JBU 放大到像素级特征，再与 9 类文本特征计算相似度并生成掩码。

## 2. 当前完成的优化

### 2.1 310B 兼容层保留的算子替换

这些改造最初是为 310B1 的算子缺失或低性能路径准备的，310P 导出和运行流程仍然继承了其中已经验证有效的部分。

| 原始路径 | 替代实现 | 原因 |
|---|---|---|
| `argmax` | `ReduceMax + Equal + Where`（310B 导出基线）或 310P 原生 `argmax` | 310B1 的 ArgMaxV2 没有高优 kernel；310P 可恢复原生路径 |
| bicubic `Resize(half_pixel)` | 预计算 depthwise `Conv2d` 核 | 310P/310B 的 ATC 对该 Resize 坐标模式不稳定或不支持 |
| `F.unfold` / Im2Col | 显式窗口切片和逐窗口累加 | ONNX 会把 Unfold 展开成 `Gather/GatherV2`，在 NPU 上产生大量低效内存访问 |
| 动态 `adaptive_avg_pool2d` | 导出时使用固定目标尺寸路径 | 静态 OM 图更容易通过 ATC shape 推导，避免动态 shape 传播失败 |
| FP32 Cube 计算 | FP16 计算或 CPU/Numpy 后处理 | 310B1 Cube 对 FP32 矩阵计算支持受限 |

310B1 项目中还验证过以下六类基础算子组合：SDPA、Linear、Linspace、ReflectionPad2d、Im2Col、Bilinear。它们位于 `ascend_310b_operators/`，主要服务 310B1 直接运行 PyTorch 的场景，不是当前 310P OM 的必需依赖。

### 2.2 310P 专用 ONNX 导出

入口：`scripts/export_onnx_310p.py`

310P 没有继续直接使用 310B 导出脚本，而是在保留 310B 基线的前提下做了单独注入：

1. **显式 Attention forward**：给 `ResidualAttentionBlock` 注入 ONNX-safe 的 Q/K/V、MatMul、Softmax 和输出投影路径。原因是 310P 上 torch_npu 的原生 MHA 在导出前会触发格式转换错误。
2. **Q/K/V 显式切片**：把 `qkv.chunk(3)` 改成三个显式张量切片，避免 CANN 9.1.1 对三输出 `Split` 图的解析问题。
3. **文本 batch 固定为 8**：`clip_text` 编译为 `text:8,77`，推理时不足 8 条进行 padding，避免动态 batch 导致 ATC reshape 推导失败。
4. **ONNX opset 元数据降为 17**：PyTorch 当前导出器可能写入 opset 18，`scripts/build_om_310p.sh` 在临时副本中降为 17，再交给 ATC；原始 ONNX 文件不被改写。
5. **JBU 单独 CPU 导出**：JBU 使用 `scripts/export_jbu_onnx_310p_cpu.py` 在 CPU 加载权重和输入导出，避免 NPU 模块迁移时 torch_npu allocator/stream 状态互相影响。Visual/Text 仍在 NPU 导出。

### 2.3 JBU 的 `F.unfold` 替换

入口：`scripts/export_jbu_onnx_310p_cpu.py`

当前默认环境变量为：

```bash
SEG310P_JBU_UNFOLD_MODE=slice
```

替换分为两处：

- guidance patch：`_slice_patches()` 使用 `K*K` 个静态窗口切片后 `stack`；
- source 加权：`_slice_weighted_sum()` 对每个窗口执行 `Slice + Mul + Add` 累加。

这样导出的 JBU 图不再包含 `Gather` 和 `Einsum`，代价是图节点数增加，但 ATC 可以编译并且实际运行显著更快。

当前生产图统计：

```text
总节点数：3228
Gather：0
Einsum：0
Slice：1540
Mul：517
Add：493
```

旧 Unfold 版本仍保留在：

```text
models/onnx_310p_unfold/
models/om_310p_unfold/
```

### 2.4 ACL OM 运行时优化

入口：`utils/session.py`、`scripts/demo_om.py`

1. **文本特征缓存**：按类别词和模板计算 MD5，生成 `query_features_*.npy`；后续 OM 推理直接加载缓存，不再重复运行 `clip_text.om`。
2. **纯 Numpy ACL 后处理**：纯 OM 路径用 Numpy 完成 L2 normalize、相似度 `einsum`、softmax、同义词类别合并和 argmax，避免在 310P/310B 上额外初始化 torch_npu。
3. **ACL context 恢复**：每次 OM execute 前调用 `acl.rt.set_context()`，避免其它 NPU 操作切换当前 ACL context 后出现全零输出或执行异常。
4. **显式资源释放**：OM、dataset、device/host buffer 和 ACL context 在 `close()` 中释放，并用 `_closed` 防止重复释放。
5. **静态输入图**：Visual、Text、JBU 均按固定 shape 编译，减少运行时 shape 推导和内存分配不确定性。

### 2.5 可视化输出

模型索引掩码不变，只新增了两套可视化：

- OM/PyTorch 原始颜色输出；
- `distinct9` 高对比九色输出，便于人工区分相邻类别。

可视化颜色不会改变模型预测或精度指标。

## 3. 当前模型产物

```text
models/onnx_310p/clip_visual.onnx
models/onnx_310p/clip_text.onnx
models/onnx_310p/jbu_upsampler.onnx

models/om_310p/clip_visual.om
models/om_310p/clip_text.om
models/om_310p/jbu_upsampler.om
models/om_310p/query_features_full_fe86ede95e67.npy
```

构建脚本：`scripts/build_om_310p.sh`  
目标 SoC：`Ascend310P3`

## 4. 为什么这样改

### 4.1 为什么不继续使用原始 Unfold

`F.unfold` 在 ONNX 中通常会变成 Gather 类节点。JBU 的输入是 512 通道、全分辨率 `224x224` 特征图，每个 crop 需要搬运大量大张量；在 310P 上该路径几乎被 Gather/内存访问占满。静态 Slice 虽然会使图变大，但可被 ATC 编译成明确的切片、逐元素乘加路径，实际吞吐远高于 Gather 展开图。

### 4.2 为什么 JBU 放到 CPU 导出

导出阶段不需要硬件执行 JBU。CPU 导出只负责用权重和示例输入追踪计算图，能够绕过 torch_npu 的设备迁移、stream 和 allocator 冲突；最终得到的 ONNX 仍由 ATC 编译为 310P OM，运行阶段不在 CPU 上执行 JBU。

### 4.3 为什么文本特征要缓存

文本特征只由类别名称、模板和 CLIP Text Encoder 决定，与输入图像无关。9 类文本展开为 14 个 query words，缓存文件只有约 14 KB，却可以避免每次启动加载 Text OM 和重复模板推理，降低显存峰值和启动时间。

### 4.4 为什么保留 310B 基线和回退目录

310B 与 310P 的 ATC 算子库不同。310P 可以恢复部分原生算子，但不代表同一图在 310B 上也能通过。因此 `scripts/export_onnx.py` 继续作为 310B 参考，310P 使用独立脚本和目录，旧 Unfold OM 也不覆盖，便于 A/B 回退。

## 5. 本次 310P 同输入实测

统一条件：

- 输入：`demo/oem_koeln_50.tif`，原图 `1000x1000`；
- 类别：同一份 9 类文本；
- 输入尺寸：`448x448`；
- crop/stride：`224/112`；
- JBU：开启，`num_levels=1`；
- `prob_thd=0.1`，`cls_token_lambda=-0.3`，`template=full`；
- OM 使用第二张物理卡；PyTorch 单独测试使用第一张物理卡；
- PyTorch 和 OM 的耗时均为模型已经加载后的单次预测耗时，不含模型加载。

### 5.1 Torch 与 OM：输入完全一致

| 后端 | 硬件 | 单次预测 | 输出尺寸 | pixel accuracy | mIoU（9 类） | 与对方掩码一致率 |
|---|---|---:|---:|---:|---:|---:|
| PyTorch + torch_npu | 310P3 | 27.97 s | 1000x1000 | 0.747165 | 0.283411 | - |
| 最新纯 OM（Slice JBU） | 310P3 | 8.60 s | 1000x1000 | 0.746837 | 0.285128 | 97.9419% |

补充：OM 与旧 Slice 生产输出逐像素一致率为 100%；PyTorch 与 OM 的差异像素为 `20,581 / 1,000,000`，主要集中在 building/road 等边界区域。当前测得 OM 约为 PyTorch 的 `3.25x`（27.97 / 8.60）。

输出位置：

```text
demo_output_310p_latest9/oem_koeln_50_mask_om_310p_demo9_latest.png
demo_output_pytorch_latest9/oem_koeln_50_mask_torch_9class.png
```

### 5.2 JBU 优化前后

| JBU 版本 | 单次整图预测 | JBU 单 crop | 与 Slice 版一致率 |
|---|---:|---:|---:|
| Unfold/Gather OM | 约 62.39 s | 约 6.45 s | 99.9731% |
| Slice OM | 8.60~9.03 s | 约 0.508 s | 100%（生产副本） |

整图耗时受进程、首次运行和 ACL 调度影响；JBU 单模块和整图趋势一致，Slice 路径是当前生产版本。

## 6. 原文档中的历史 310B/910 数据

下面数据来自已有文档，测试边界与本次 310P 不完全相同，因此不能直接当作同输入横向对比。

### 6.1 310B1 Level-1 PyTorch patch A/B

来源：`ascend_310b_operators/scaled_dot_product_attention/evidence/LEVEL1_PATCH_COMPARISON_AND_OPERATOR_PERF_20260824.md`

| 配置 | 输入/模型条件 | 墙钟时间 | 输出一致性 |
|---|---|---:|---|
| `USE_PATCH=True` | 310B1，`num_levels=1`，完整 demo | 92.135 s | 与 patch-off 完全一致 |
| `USE_PATCH=False` | 同一 310B1 输入、权重和类别 | 97.110 s | 与 patch-on 完全一致 |

该文档只说明 A/B 使用同一输入、权重和类别，没有给出本次 `demo/oem_koeln_50.tif` 的文件名、当前 9 类配置和当前 OM 对应的统一计时范围，因此只能作为 310B1 patch 开关参考。

### 6.2 310B1 热态 PyTorch

同一份 Level-1 文档记录：

| 配置 | 模型构建 | 热态单次 Median | 备注 |
|---|---:|---:|---|
| Patch-Off | 19.7885 s | 25.2589 s | `num_levels=1`，模型常驻后测量 |
| Patch-On | 20.5162 s | 25.5437 s | 同一测试脚本 |

该数据是 310B1 的 PyTorch 热态数据，不是当前 310P 的 `27.97 s`；硬件、算子路径、计时脚本和文档版本均不同。

### 6.3 310B1 纯 ACL 理论/Profiling 数据

来源：`docs/theoretical_throughput_report.md`。

| 指标 | 310B1 历史值 | 测试条件 |
|---|---:|---|
| ACL 实测整图时间 | 24.28 s/image | 6 crops，`eval_acl.py`，1 image profiling 工作负载 |
| 实测吞吐量 | 0.04 img/s | 同上 |
| JBU 占 per-crop | 约 90% | 旧 Gather/大内存搬运图 |
| 理论 memory-bound 上限 | 9.416 s/image | 仅理论带宽估算，不是实测 |

该数据使用 310B1、6 crop 和历史 ACL 评测流程，与本次 310P 的 448 输入 9 类单图结果不一致，不能直接用来计算 310P 加速比。

### 6.4 原部署文档中的 UDD5 精度数据

来源：`docs/deployment.md`。这是历史 UDD5 全量 40 张评测，类别、数据集和输入集合都不同；文档环境主要记录为 Ascend 910_9382，不能标记为本次 310P 数据。

| 后端 | 数据集/样本 | aAcc | mIoU | mAcc |
|---|---|---:|---:|---:|
| PyTorch | UDD5，40 张 | 74.86% | 50.55% | 66.15% |
| ACL OM | UDD5，40 张 | 74.87% | 50.56% | 66.16% |

## 7. 数据可比性结论

1. **可以直接比较的有第 5.1 节和第 9.4 节**：第 5.1 节是单张 OpenEarthMap 的 Torch/OM 对比；第 9.4 节是当前最新 310P OM 的 UDD5 全量评测。
2. **310B1 的历史时间不能与本次 310P 时间直接做硬件加速比**：至少存在硬件型号、crop 数、数据集、模板、脚本和计时范围差异。
3. **310B1 Level-1 patch A/B 可以在内部比较**，因为它们使用同一输入和同一权重；它只说明 patch-on/off 的相对变化。
4. **UDD5 的历史 50.55/50.56 mIoU 与本次最新 310P 的 50.51 mIoU 可以作为同数据集参考，但评测后端、模型版本和配置仍需区分**；不能与当前 OpenEarthMap `koeln_50` 的 0.283/0.285 mIoU 混用。
5. 如果需要严格的 310B 与 310P 横向性能表，应在两台设备上使用同一张 `oem_koeln_50.tif`、同一 9 类文件、同一 `448/224/112` 滑窗配置，并分别记录 warm-up、模型加载和单次 predict 三个时间口径。

## 8. 复现命令

```bash
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate SegEarth310
source <CANN_ROOT>/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=1

cd SegEarth-OV
python scripts/demo_om.py \
  -i demo/oem_koeln_50.tif \
  -o demo_output_310p_latest9 \
  --config configs/cfg_demo9_om.py \
  --om-dir models/om_310p \
  --device-id 0 \
  --size 448 \
  --template full \
  --prob-thd 0.1
```

PyTorch 直接推理需要确保 9 类文件是真正的九行文本，并使用未被其它进程占用的 NPU；OM 路径则优先使用缓存的 `query_features_full_*.npy`。

## 8.1 ONNX 导出与 OM 编译命令

### 最新 OM 目录

当前生产目录为：

```text
SegEarth-OV/models/om_310p
```

其中主要产物为：

```text
clip_visual.om
clip_text.om
jbu_upsampler.om
query_features_full_fe86ede95e67.npy
query_features_sub_8acc9ae29c5e.npy
```

对应 ONNX 目录为：

```text
SegEarth-OV/models/onnx_310p
```

### 重新导出 310P ONNX

该命令会在 NPU 上导出 `clip_visual` 和 `clip_text`，并在同一脚本中启动独立 CPU 进程导出 JBU。默认输出到 `models/onnx_310p`，默认使用 Slice JBU 路径。

```bash
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate SegEarth310
source <CANN_ROOT>/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=1

cd SegEarth-OV
export SEG310P_OUTPUT_DIR=models/onnx_310p
export SEG310P_JBU_UNFOLD_MODE=slice
python scripts/export_onnx_310p.py
```

如果需要导出到临时目录，修改 `SEG310P_OUTPUT_DIR` 即可；不要直接覆盖 `models/onnx_310p_unfold` 回退目录。

### 编译 310P OM

`scripts/build_om_310p.sh` 会把 ONNX 临时转换为 ATC 可接受的 opset 17 元数据，再分别编译 Visual、Text 和 JBU。临时转换不会修改原始 ONNX。

```bash
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate SegEarth310
source <CANN_ROOT>/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=1

cd SegEarth-OV
export ONNX_DIR=models/onnx_310p
export OM_DIR=models/om_310p
export SOC_VERSION=Ascend310P3
bash scripts/build_om_310p.sh
```

编译日志位于：

```text
SegEarth-OV/logs/om_310p/
```

编译完成后可检查：

```bash
ls -lh SegEarth-OV/models/om_310p/*.om
```

### 使用最新 OM 推理

```bash
python scripts/demo_om.py \
  -i demo/oem_koeln_50.tif \
  -o demo_output_310p_latest9 \
  --config configs/cfg_demo9_om.py \
  --om-dir models/om_310p \
  --device-id 0 \
  --size 448 \
  --template full \
  --prob-thd 0.1
```

## 9. 最终指标汇总

### 9.1 310P：同一输入、同一 9 类配置

| 后端 | 输入 | 类别 | 单次预测 | pixel accuracy | mIoU | 与对方掩码一致率 | 是否可直接比较 |
|---|---|---|---:|---:|---:|---:|---|
| PyTorch + torch_npu | `demo/oem_koeln_50.tif` | 9 类 | 27.97 s | 0.747165 | 0.283411 | 97.9419% | 是 |
| 310P 纯 OM（Slice JBU） | `demo/oem_koeln_50.tif` | 9 类 | 8.60 s | 0.746837 | 0.285128 | 97.9419% | 是 |

统一配置为 `448/224/112`（输入/crop/stride）、JBU `num_levels=1`、`template=full`、`prob_thd=0.1`、`cls_token_lambda=-0.3`。两侧均为模型加载后的单次预测时间，不含模型加载；OM 约为 PyTorch 的 `3.25x`。

### 9.2 310P：JBU Unfold 与 Slice

| JBU 图 | 整图单次预测 | JBU 单 crop | 图特征 | 精度/一致性 |
|---|---:|---:|---|---|
| Unfold/Gather OM | 约 62.39 s | 约 6.45 s | `Gather=16`、`Einsum=4` | 与 Slice 像素一致率 99.9731% |
| Slice OM（当前生产） | 8.60~9.03 s | 约 0.508 s | `Gather=0`、`Einsum=0`、`Slice=1540` | 生产副本一致率 100% |

### 9.3 历史 310B1/910 数据：不可与 9.1 直接横比

| 平台/后端 | 输入和配置 | 性能/精度指标 | 数据性质 |
|---|---|---|---|
| 310B1 PyTorch patch-on | Level-1 历史 demo，`num_levels=1` | 92.135 s 墙钟 | 与 patch-off 同输入 A/B，可内部比较 |
| 310B1 PyTorch patch-off | 同上 | 97.110 s 墙钟；热态 Median 25.2589 s | 与 patch-on 同输入 A/B |
| 310B1 纯 ACL | 历史 `eval_acl.py`，6 crops，1 image | 24.28 s/image；0.04 img/s | 不是当前 310P 输入/滑窗配置 |
| UDD5 PyTorch | 历史 40 张，文档环境为 910_9382 | mIoU 50.55%，aAcc 74.86% | 数据集和类别不同 |
| UDD5 ACL OM | 同上 | mIoU 50.56%，aAcc 74.87% | 数据集和类别不同 |

因此，表 9.1 是严格同输入的 310P Torch/OM 单图对比，表 9.4 是最新 310P OM 的 UDD5 全量结果；历史 310B 数据已列出，但不能直接用于计算 310B 到 310P 的硬件加速比。

### 9.4 最新 310P OM：UDD5 全量验证集

本次评测使用当前生产目录 `models/om_310p` 中的 Slice JBU OM，UDD5 验证集共 40 张，配置为 `cfg_udd5.py`、5 类文本、`template=full`、输入尺寸 `448`、JBU 开启，运行在物理第二张 310P（进程内 `device-id=0`）。

| 后端/模型 | 样本数 | aAcc | mIoU | mAcc | 平均推理 | 总推理 | 吞吐量 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 最新 310P Slice OM | 40 | 74.82% | 50.51% | 66.10% | 6.667 s/img | 266.67 s | 0.15 img/s |

逐类 IoU：vegetation `80.54%`，building `70.87%`，road `53.27%`，vehicle `21.52%`，background `26.37%`。

评测日志：

```text
SegEarth-OV/logs/eval_310p_udd5_latest/eval.log
```

这次结果和历史文档中的 UDD5 ACL `50.56%` mIoU 相差 `0.05` 个百分点；历史结果的具体 OM 目录、硬件和完整命令口径与当前 310P 不完全相同，因此这里保留为参考，不把两者合并成同一条实验记录。

### 9.5 最新 310P PyTorch：UDD5 全量验证集

PyTorch 使用与 OM 相同的 `cfg_udd5.py`、5 类文本、`template=full`、输入尺寸 `448` 和 JBU 配置；另外必须先执行 `setup_npu(use_patch=False, use_interpolate_patch=False)`。如果直接执行 `--backend pytorch` 而不初始化该配置，JBU 的 `Im2col` 可能在 310P 上因显存分配失败而 OOM。

| 后端/模型 | 样本数 | aAcc | mIoU | mAcc | 平均推理 | 总推理 | 吞吐量 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 最新 310P PyTorch（patch-off） | 40 | 74.62% | 50.38% | 65.89% | 14.226 s/img | 569.05 s | 0.07 img/s |

逐类 IoU：vegetation `80.42%`，building `70.55%`，road `53.09%`，vehicle `21.50%`，background `26.33%`。

评测日志：

```text
SegEarth-OV/logs/eval_310p_udd5_pytorch_patchoff/eval.log
```

### 9.6 最新 310P UDD5 Torch/OM 对比

| 后端 | aAcc | mIoU | mAcc | 平均推理 | 吞吐量 |
|---|---:|---:|---:|---:|---:|
| PyTorch（patch-off） | 74.62% | 50.38% | 65.89% | 14.226 s/img | 0.07 img/s |
| Slice OM | 74.82% | 50.51% | 66.10% | 6.667 s/img | 0.15 img/s |
| OM - PyTorch | +0.20 个百分点 | +0.13 个百分点 | +0.21 个百分点 | -7.559 s/img | +0.08 img/s |

在这次 UDD5 全量口径下，OM 单张推理约为 PyTorch 的 `2.13x`；两者使用同一验证集和类别配置，但 PyTorch 与 OM 的 JBU/数值路径不同，因此指标不要求逐像素完全一致。

#### 310B1 历史 UDD5 基准

项目历史 README/部署记录还保存了一组 UDD5 验证集全量 40 张的 310B/ACL 参考指标。该表可作为精度基准，但没有与当前 310P 相同的单张推理计时记录：

| 历史后端 | 样本数 | aAcc | mIoU | mAcc | 与当前 310P 的关系 |
|---|---:|---:|---:|---:|---|
| 310B1 PyTorch / eval.py | 40 | 74.86% | 50.55% | 66.15% | 历史同数据集精度基准 |
| 310B1 ACL/OM | 40 | 74.87% | 50.56% | 66.16% | 历史同数据集 OM 精度基准 |

当前 310P 相对这组历史基准：

| 对比 | aAcc | mIoU | mAcc |
|---|---:|---:|---:|
| 310P PyTorch - 310B1 PyTorch | -0.24 个百分点 | -0.17 个百分点 | -0.26 个百分点 |
| 310P Slice OM - 310B1 ACL/OM | -0.05 个百分点 | -0.05 个百分点 | -0.06 个百分点 |

这里的“310B1”标签来自项目历史 README 和 `run_310b1.sh` 评测记录；`docs/deployment.md` 同时保留了 910 交叉编译环境说明，因此这组旧指标只作为项目历史基准，不作为严格硬件 A/B 结论。历史 310B1 ACL 的性能记录是另一套 6-crop profiling 口径（约 24.28 s/image），与当前 UDD5 40 张、448 输入的 310P 表格不能直接比较。

## 10. UDD5 评测命令

### 10.1 最新 OM

```bash
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate SegEarth310
source <CANN_ROOT>/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=1
cd SegEarth-OV

python scripts/eval_acl.py \
  --config configs/cfg_udd5.py \
  --backend acl \
  --om-dir models/om_310p \
  --device-id 0 \
  --size 448 \
  --max-samples 0 \
  --template full
```

### 10.2 最新 PyTorch（310P patch-off）

```bash
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate SegEarth310
source <CANN_ROOT>/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
cd SegEarth-OV

python -u -c 'import runpy,sys; from npu_compat import setup_npu; setup_npu(use_patch=False,use_interpolate_patch=False); sys.argv=["scripts/eval_acl.py","--config","configs/cfg_udd5.py","--backend","pytorch","--device-id","0","--size","448","--max-samples","0","--template","full"]; runpy.run_path("scripts/eval_acl.py",run_name="__main__")'
```

这里 `ASCEND_RT_VISIBLE_DEVICES=1` 表示使用物理第二张卡，进程内设备号仍然是 `0`。如果使用其他物理卡，只需修改该环境变量，保留 `--device-id 0`。
