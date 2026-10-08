# 310P 远程仓代码变更梳理

> 记录日期：2026-09-29  
> 远程主机：`310p_169`  
> 远程项目：`SegEarth-OV`  
> 目标设备：Ascend 310P3  
> 运行环境：Conda `SegEarth310`，CANN：`<CANN_ROOT>/set_env.sh`

## 1. 范围说明

本文档只记录本次远程 310P 导出、编译、运行和 UDD5 验证流程中明确使用的代码与运行时改动。远程仓当前本身包含较多历史性的 310B/ACL 改动；因此下面将内容分成两类：

1. **本次 310P 专用新增文件**：用于生成 310P ONNX/OM 和 9 类 demo。
2. **此前已有、被 310P 流程继承使用的适配文件**：主要是 310B/ACL 运行时改造，不把它们误报为本次新增。

远程 `git status` 显示本次 310P 专用文件目前是未跟踪文件（`??`），而继承文件显示为历史修改（`M`）。本文档是按功能划分的交接记录，不是完整的 Git 归属报告。

## 2. 本次新增的 310P 专用代码

### 2.1 `scripts/export_onnx_310p.py`

这是独立的 310P ONNX 导出入口，保留 `scripts/export_onnx.py` 作为 310B 基线，不直接覆盖 310B 输出。

| 行号 | 变更 | 目的 |
|---|---|---|
| 38-43 | 从 CLIP 导出进程中移除 JBU 导出调用 | Visual/Text 仍在 NPU 导出，JBU 改由独立 CPU 进程导出，避免 torch_npu 的 allocator/stream 状态冲突 |
| 45-85 | 注入 `_onnx_safe_attention` 并替换 `ResidualAttentionBlock.attention` | 绕过 310P 上原生 MHA 在导出前的格式转换错误，显式生成 Q/K/V、MatMul、Softmax 和输出投影 |
| 87-95 | 310P 恢复原生 `x.argmax(dim=-1)` | 310P 可使用原生 ArgMax，避免 310B 基线的 `ReduceMax + Equal + Where` 替代图 |
| 97-110 | 将输出目录改为 `models/onnx_310p`，支持 `SEG310P_OUTPUT_DIR` | 与 310B ONNX 产物隔离，并支持临时 A/B 输出目录 |
| 112-121 | 保留深度卷积形式的 bicubic 上采样；将 guidance 的固定尺寸路径改为 `F.interpolate(..., bilinear)` | CANN 对 `Resize(bicubic, half_pixel)` 编译不稳定；固定输入下保留可编译路径 |
| 122-131 | Text 导出使用静态 batch 8，并关闭动态轴 | 与 ACL 运行时 padding 到 8 的输入约定一致，避免 ATC 动态 shape 推导失败 |
| 132-142 | 将 `qkv.chunk(3)` 改为三个显式切片 | 避免 CANN 9.1.1 解析三输出 `Split` 节点失败 |
| 155-159 | 导出完成后调用 `export_jbu_onnx_310p_cpu.py` | 一条命令同时生成 Visual/Text/JBU 三个 310P ONNX，JBU 实际在 CPU 导出 |

默认输出：`models/onnx_310p/{clip_visual,clip_text,jbu_upsampler}.onnx`。

### 2.2 `scripts/export_jbu_onnx_310p_cpu.py`

这是隔离的 CPU JBU ONNX 导出器，加载 CPU 上的权重和示例输入；CPU 只负责追踪计算图，最终图仍由 ATC 编译并在 310P 上执行。

| 行号 | 变更 | 目的 |
|---|---|---|
| 25-31 | `_slice_patches()` 用静态窗口切片和 `stack` 提取 guidance patch | 替代 `F.unfold` 导出的 Gather/Im2Col 路径 |
| 34-43 | `_slice_weighted_sum()` 用 Slice + Mul + Add 逐窗口累加 | 替代 JBU 末端的 `Einsum`，让 ATC 获得明确的逐元素图 |
| 55-117 | 重写 `JBULearnedRange.forward`，默认 `SEG310P_JBU_UNFOLD_MODE=slice` | 同时覆盖 guidance patch 和 source 加权两处 Unfold；保留 `unfold` 回退模式便于 A/B |
| 92-117 | 保留四个 depthwise-conv bicubic 核；Slice 模式使用 `_slice_weighted_sum` | 绕过 310P 对 bicubic/half_pixel Resize 的编译限制 |
| 120-123 | `_pool_step()` 保持与 `JBUOne.upsample` 相同的 adaptive average pooling 语义 | 确保导出图和 PyTorch JBU 的数值路径一致 |
| 132-164 | CPU 注入 bicubic buffer，FP16 CPU wrapper 导出固定输入输出 | 导出接口保持 ACL/OM 运行时所需的 FP16 和静态 shape |

生产 Slice 图的统计：`nodes=3228`、`Gather=0`、`Einsum=0`、`Slice=1540`、`Mul=517`、`Add=493`。旧 Unfold 版本保留在 `models/onnx_310p_unfold/` 和 `models/om_310p_unfold/`。

### 2.3 `scripts/build_om_310p.sh`

这是 310P 的 ATC 构建脚本，默认使用 `Ascend310P3`。

| 行号 | 变更 | 目的 |
|---|---|---|
| 12-15 | 默认 ONNX=`models/onnx_310p`、OM=`models/om_310p`、日志=`logs/om_310p`、SoC=`Ascend310P3` | 固定 310P 生产目录，同时允许环境变量覆盖 |
| 17-19 | 使用临时目录并在退出时清理 | 不改写原始 ONNX，隔离 ATC 中间文件 |
| 28-45 | 用 `onnx` 将标准域 opset 元数据临时降为 17 | PyTorch 导出器可能写入 opset 18，而当前 CANN 9.1.1/310P parser 对该图按 opset 17 才能通过 |
| 47-58 | 封装 `compile_model()` 并保存每个模型的 ATC 日志 | 统一构建参数和问题定位入口 |
| 61-63 | 固定编译输入：Visual `1,3,224,224`；Text `8,77`；JBU `1,512,14,14 + 1,3,224,224` | 与运行时静态输入约定一致 |

### 2.4 `configs/cfg_demo9_om.py`

基于 `cfg_udd5.py` 的 9 类 OM demo 配置（第 1-8 行）：类别文件为 `configs/cls_demo9.txt`，并设置 `prob_thd=0.1`、`bg_idx=0`、`cls_token_lambda=-0.3`。

### 2.5 `configs/cls_demo9.txt`

第 1-9 行定义当前一致性比较使用的 9 类及同义词：

```text
background
bareland,barren
grass
pavement
road
tree,forest
water,river
cropland
building,roof,house
```

## 3. 继承使用的 310B/ACL 适配代码

以下文件在远程仓中已经存在历史修改，本次 310P 流程继续调用它们的接口；它们不是本次 310P 专用文件，但没有这些适配，OM demo 或评测无法按当前方式运行。

| 文件 | 关键位置 | 在当前流程中的作用 |
|---|---|---|
| `utils/session.py` | `Session`/`AclSession`，约第 55-85、394-430、372 行 | ACL context 恢复；按类别和模板缓存 query feature；Numpy 后处理；OM、dataset、buffer 资源释放 |
| `scripts/demo_om.py` | `main()`，约第 154-234 行 | 单图 OM 推理、滑窗拼接、掩码和彩色输出 |
| `scripts/eval_acl.py` | `main()`，约第 174-278 行 | UDD5 的 ACL/PyTorch 评测和 aAcc/mIoU/mAcc 统计 |
| `scripts/demo_multi.py` | `main()`，约第 61-110 行 | 多后端 demo 入口 |
| `scripts/benchmark_om.py` | `benchmark_one()`/`main()`，约第 99-244 行 | warm-up、重复推理和后端耗时统计 |
| `scripts/export_onnx.py` | JBU 导出、Visual/Text 导出函数，约第 109-162、295-479 行 | 310P exporter 的基线源码和 bicubic buffer 注入来源 |
| `segearth_segmentor.py` | query feature、滑窗预测和类别合并，约第 150-235、342-345 行 | PyTorch 路径的文本特征和掩码逻辑基准 |
| `npu_compat.py` | `setup_npu()` 约第 250 行 | 310B/310P 的 patch 开关、SDPA/插值兼容控制；PyTorch patch-off 评测使用 `use_patch=False, use_interpolate_patch=False` |
| `simfeatup_dev/upsamplers.py` | `JBULearnedRange`/`JBUOne`，约第 202-328 行 | JBU 权重结构和 PyTorch 语义来源 |
| `ascend_310b_operators/` | SDPA、Linear、Linspace、ReflectionPad2d、Im2Col、Bilinear 六类目录 | 310B 兼容实现和证据；310P OM 生产图不是这些 Python patch 的直接运行路径 |

### 3.1 继承适配中与 310P 相关的算子结论

- 310B 基线的 `argmax` 替代图在 310P 版本恢复为原生 `argmax`。
- Bicubic `Resize(half_pixel)` 仍不能直接交给当前 ATC，因此 310P 继续使用预计算 depthwise-conv 核。
- JBU 的两处 `F.unfold`/`Einsum` 已在 310P 专用 CPU exporter 中改成 Slice/Mul/Add。
- 310B 的 SDPA、Linear、Linspace、ReflectionPad2d、Im2Col、Bilinear 目录主要用于 Python/NPU 兼容验证，不应写成 310P OM 图中必然加载的算子库。

## 4. 运行时和数据侧变更

### 4.1 UDD 数据软链接

创建了：

```text
SegEarth-OV/data/UDD/UDD
  -> <UDD_DATA_ROOT>
```

这样 `configs/cfg_udd5.py` 中的路径可以解析到：

```text
data/UDD/UDD/UDD5/val/src
data/UDD/UDD/UDD5/val/gt
```

当前 UDD5 验证集为 40 对图像/GT，训练集为 120 对图像/GT。

### 4.2 当前模型产物

生产目录：

```text
SegEarth-OV/models/onnx_310p
SegEarth-OV/models/om_310p
```

生产 OM 文件：

```text
clip_visual.om
clip_text.om
jbu_upsampler.om
query_features_full_fe86ede95e67.npy
query_features_sub_8acc9ae29c5e.npy
```

对应的编译/评测日志：

```text
logs/om_310p/clip_visual.log
logs/om_310p/clip_text.log
logs/om_310p/jbu_upsampler.log
logs/eval_310p_udd5_latest/eval.log
logs/eval_310p_udd5_pytorch_patchoff/eval.log
```

## 5. 已验证结果（用于确认代码变更生效）

### 5.1 UDD5 全量 40 张

| 后端 | aAcc | mIoU | mAcc | 平均耗时 |
|---|---:|---:|---:|---:|
| 310P 最新 Slice OM | 74.82% | 50.51% | 66.10% | 6.667 s/image |
| 310P PyTorch，patch/interpolate patch 关闭 | 74.62% | 50.38% | 65.89% | 14.226 s/image |

OM 相对 PyTorch：aAcc `+0.20` 个百分点，mIoU `+0.13` 个百分点，mAcc `+0.21` 个百分点；平均单图耗时减少 `7.559 s`，约 `2.13x`。

### 5.2 单图 9 类一致性检查

输入：`demo/oem_koeln_50.tif`，同一份 9 类文本、`448/224/112` 滑窗配置。

| 后端 | 单次耗时 | pixel accuracy | mIoU | 掩码一致率 |
|---|---:|---:|---:|---:|
| PyTorch + torch_npu | 27.97 s | 0.747165 | 0.283411 | - |
| 最新 Slice OM | 8.60 s | 0.746837 | 0.285128 | 97.9419% |

输出示例：

```text
demo_output_310p_latest9/oem_koeln_50_mask_om_310p_demo9_latest.png
demo_output_pytorch_latest9/oem_koeln_50_mask_torch_9class.png
```

## 6. 复现入口

```bash
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate SegEarth310
source <CANN_ROOT>/set_env.sh
cd SegEarth-OV

# 生成 310P ONNX；Visual/Text 在 NPU，JBU 在 CPU
python scripts/export_onnx_310p.py

# 编译 310P OM
bash scripts/build_om_310p.sh

# 9 类 OM 单图 demo
ASCEND_RT_VISIBLE_DEVICES=1 python scripts/demo_om.py \
  --config configs/cfg_demo9_om.py \
  --input demo/oem_koeln_50.tif \
  --om-dir models/om_310p

# UDD5 OM 评测
ASCEND_RT_VISIBLE_DEVICES=1 python scripts/eval_acl.py \
  --config configs/cfg_demo9_om.py \
  --om-dir models/om_310p
```

## 7. 注意事项

1. 远程工作区包含大量历史 310B/ACL 文件修改；不要用整份 `git diff` 将所有变更都归因于本次 310P 工作。
2. `models/onnx_310p_unfold/` 和 `models/om_310p_unfold/` 是旧版回退产物，生产版本是 Slice 目录。
3. 当前评测结果来自相同 9 类和相同 UDD5 验证集，但历史 310B/910 文档中的耗时与本次 310P 不一定使用相同输入、crop 数和计时口径，不能直接做硬件横向结论。
4. 若重新导出或编译，先 source CANN 环境；若只运行 OM，仍需保证 `ASCEND_RT_VISIBLE_DEVICES` 指向空闲卡并使用对应 `models/om_310p` 目录。

相关的完整优化和性能说明见：`docs/310P_OPTIMIZATION_SUMMARY.md`。
