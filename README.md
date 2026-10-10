<div align="center">

<h1>SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images</h1>

<h3>✨CVPR 2025 Oral✨</h3>

<div>
    <strong>Make OVSS possible in remote sensing contexts</strong>
</div>

<div>
    <a href='https://likyoo.github.io/' target='_blank'>Kaiyu Li</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=WTleRV8AAAAJ' target='_blank'>Ruixun Liu</a><sup>1</sup>&emsp;
    <a href='https://gr.xjtu.edu.cn/en/web/caoxiangyong' target='_blank'>Xiangyong Cao</a><sup>✉1</sup>&emsp;
    <a href='https://web.xidian.edu.cn/xrbai' target='_blank'>Xueru Bai</a><sup>2</sup>&emsp;
    <a href='https://faculty.xidian.edu.cn/ZF3' target='_blank'>Feng Zhou</a><sup>2</sup>&emsp;
    <a href='https://gr.xjtu.edu.cn/en/web/dymeng' target='_blank'>Deyu Meng</a><sup>1</sup>&emsp;
    <a href='https://gr.xjtu.edu.cn/en/web/zhiwang' target='_blank'>Zhi Wang</a><sup>1</sup>&emsp;
</div>
<div>
    <sup>1</sup>Xi'an Jiaotong University&emsp;
    <sup>2</sup>Xidian University&emsp;
</div>

<div>
    <h4 align="center">
        • <a href="https://likyoo.github.io/SegEarth-OV/" target='_blank'>[Project]</a> • <a href="https://arxiv.org/abs/2410.01768" target='_blank'>[arXiv]</a> • <a href="https://colab.research.google.com/drive/1a-NNz_2maesvszk4Xff5PKY02_moPqt6#scrollTo=Pz9QGEcFBGtK" target='_blank'>[Colab]</a> •
    </h4>
</div>

<img src="https://github.com/user-attachments/assets/28675180-02de-476e-ad01-7c2138a2a943" width="100%"/>
Visualization and performance of SegEarth-OV on open-vocabulary semantic segmentation of remote sensing images. We evaluate on 17 remote sensing datasets (including semantic segmentation, building extraction, road extraction, and flood detection tasks), and our SegEarth-OV consistently generates high-quality segmentation masks.

</div>

## Abstract
> *Remote sensing image plays an irreplaceable role in fields such as agriculture, water resources, military, and disaster relief. Pixel-level interpretation is a critical aspect of remote sensing image applications; however, a prevalent limitation remains the need for extensive manual annotation. For this, we try to introduce open-vocabulary semantic segmentation (OVSS) into the remote sensing context. However, due to the sensitivity of remote sensing images to low-resolution features, distorted target shapes and ill-fitting boundaries are exhibited in the prediction mask. To tackle this issue, we propose a simple and general upsampler, SimFeatUp, to restore lost spatial information in deep features in a training-free style. Further, based on the observation of the abnormal response of local patch tokens to [CLS] token in CLIP, we propose to execute a straightforward subtraction operation to alleviate the global bias in patch tokens. Extensive experiments are conducted on 17 remote sensing datasets spanning semantic segmentation, building extraction, road detection, and flood detection tasks. Our method achieves an average of 5.8%, 8.2%, 4%, and 15.3% improvement over state-of-the-art methods on 4 tasks. All codes are released.*

## Dependencies and Installation


```
# 1. install SimFeatUp
# refer to https://github.com/likyoo/SimFeatUp

# 2. git clone this repository
git clone https://github.com/likyoo/SegEarth-OV.git
cd SegEarth-OV

# 3. create new anaconda env
conda create -n SegEarth python=3.9
conda activate SegEarth

# install torch and dependencies
pip install -r requirements.txt
# The dependent versions are not strict, and in general you only need to pay attention to mmcv and mmsegmentation.
```


## Datasets
We include the following dataset configurations in this repo: 
1) `Semantic Segmentation`: OpenEarthMap, LoveDA, iSAID, Potsdam, Vaihingen, UAVid<sup>img</sup>, UDD5, VDD
2) `Building Extraction`: WHU<sup>Aerial</sup>, WHU<sup>Sat.Ⅱ</sup>, Inria, xBD<sup>pre</sup>
4) `Road Extraction`: CHN6-CUG, DeepGlobe, Massachusetts, SpaceNet
5) `Water Extraction`: WBS-SI

Please refer to [dataset_prepare.md](https://github.com/likyoo/SegEarth-OV/blob/main/dataset_prepare.md) for dataset preparation.


## Quick Inference
```
python demo.py
```

## Model evaluation
Single-GPU:

```
python eval.py --config ./configs/cfg_DATASET.py --workdir YOUR_WORK_DIR
```

Multi-GPU:
```
bash ./dist_test.sh ./config/cfg_DATASET.py
```

Evaluation on all datasets:
```
python eval_all.py
```
Results will be saved in `results.xlsx`.

## Results

<div>
<img src="https://github.com/user-attachments/assets/445afb44-447b-4aa2-b6e7-c2e7bf2b5450" width="100%"/>
</div>

<div>
<img src="https://github.com/user-attachments/assets/7270efea-2c8e-485c-a47c-48be8718000f" width="100%"/>
</div>

## Citation

```
@inproceedings{li2025segearthov,
  title={Segearth-ov: Towards training-free open-vocabulary segmentation for remote sensing images},
  author={Li, Kaiyu and Liu, Ruixun and Cao, Xiangyong and Bai, Xueru and Zhou, Feng and Meng, Deyu and Wang, Zhi},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={10545--10556},
  year={2025}
}
```

## Acknowledgement
This implementation is based on [ClearCLIP](https://github.com/mc-lan/ClearCLIP) and [FeatUp](https://github.com/mhamilton723/FeatUp). Thanks for the awesome work.

---

# NPU Deployment

本仓库在原始 SegEarth-OV 基础上完成了昇腾 NPU 全适配，支持 **PyTorch → ONNX → OM (ATC) → ACL** 完整离线推理部署。

## 特性

- 昇腾 910 / 310B1 NPU 全适配
- 支持 PyTorch / ONNX / ACL 三后端统一评估（`--backend` 一键切换）
- OM 离线推理精度与 PyTorch 完全对齐（mIoU 差异 ≤ 0.05）
- 310B1 算子兼容层（6 个不支持的 CUDA 算子用纯 NPU 基础算子等价替代）

## NPU 环境要求

| 项目 | 版本 |
|------|------|
| 硬件 | Ascend 910 (推理+编译) / Ascend 310B1 (推理) |
| CANN | 9.0.0 |
| PyTorch | 2.8.0 |
| torch_npu | 2.8.0.post4 |
| Python | 3.10 |

### 310P 当前验证环境

下面是本仓库在 Ascend 310P3 上实际导出 ONNX、编译 OM 并运行 demo 验证过的环境。`torch`、`torch-npu` 和 `torchvision` 必须使用相互匹配的版本，不建议分别使用无上限的 `>=`。

| 项目 | 当前验证版本/路径 |
|------|------|
| Conda 环境 | `SegEarth310` |
| Python | `3.10.21` |
| PyTorch | `2.12.0+cpu` |
| torch-npu | `2.12.0.post2` |
| torchvision | `0.27.0+cpu` |
| CANN | `9.1.1` |
| CANN 环境脚本 | `<CANN_ROOT>/set_env.sh` |
| SoC | `Ascend310P3` |
| ONNX | `1.23.0` |
| ONNXScript | `0.7.2` |
| mmcv | `2.1.0` |
| mmengine | `0.10.7` |
| mmsegmentation | `1.2.2` |
| NumPy | `2.2.6` |
| timm | `1.0.30` |

初始化环境：

```bash
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate SegEarth310
source <CANN_ROOT>/set_env.sh
```

运行 310P 任务前指定空闲 NPU。以下示例使用第二张物理卡：

```bash
export ASCEND_RT_VISIBLE_DEVICES=1
```

当前 310P 模型目录约定：

```text
models/onnx_310p/    # clip_visual.onnx、clip_text.onnx、jbu_upsampler.onnx
models/om_310p/      # clip_visual.om、clip_text.om、jbu_upsampler.om
```

如果使用当前 310P 专用导出脚本，Visual/Text 在 NPU 导出，JBU 在 CPU 导出：

```bash
export SEG310P_OUTPUT_DIR=models/onnx_310p
export SEG310P_JBU_UNFOLD_MODE=slice
python scripts/export_onnx_310p.py
```

使用 Ascend310P3 编译 OM：

```bash
export ONNX_DIR=models/onnx_310p
export OM_DIR=models/om_310p
export SOC_VERSION=Ascend310P3
bash scripts/build_om_310p.sh
```

> 说明：CANN 不属于 pip requirements；每个新终端都需要先 source 上面的 CANN 环境脚本。仓库内的 `open_clip/` 是本地源码，不需要额外安装 `open-clip-torch`。

## NPU 安装

```bash
# 1. 创建 Conda 环境
conda create -n SegEarth python=3.10
conda activate SegEarth

# 2. 安装依赖
pip install -r requirements_npu.txt

# 3. 初始化 CANN 环境（每次新终端执行）
source <CANN_ROOT>/set_env.sh
```

## PyTorch 推理 (NPU)

```bash
python eval.py --config configs/cfg_udd5.py
```

## OM 离线推理部署

### 流程概览

```
PyTorch 模型 (NPU fp16)
    ├─ Visual Encoder ──→ ONNX ──→ ATC ──→ clip_visual.om
    ├─ Text Encoder   ──→ ONNX ──→ ATC ──→ clip_text.om (结果缓存为 .npy)
    └─ JBU Upsampler  ──→ ONNX ──→ ATC ──→ jbu_upsampler.om

ACL 推理:
    图像 → 滑窗 crop → visual.om → upsampler.om → 与文本特征 einsum → 分割结果
```

### Step 0: PyTorch 基线评估

```bash
bash scripts/step0_eval_pytorch.sh
```

使用原始 `eval.py`（MMEngine Runner）在 NPU 上跑 PyTorch 推理，作为精度基线。

### Step 1: ONNX 导出

```bash
bash scripts/step1_export_onnx.sh
```

使用 `scripts/export_onnx.py`（310B 优化版）导出三个子模型。相比基础版做了三个等价替代：

| 原始算子 | 问题 | 替代方案 |
|---------|------|---------|
| `argmax` | 310B1 无高优 ArgMaxV2 kernel | ReduceMax + Equal + Where |
| `F.interpolate(bicubic)` | ATC 不支持 Resize half_pixel | depthwise conv2d 等价实现 |
| `F.unfold` | ATC GatherV2 极慢 | K² 次 slice + mul + add |

### Step 2: ATC 编译 OM

```bash
# 910 上编译
bash scripts/step2_build_om.sh

# 310B1 交叉编译（在 910 上执行）
SOC_VERSION=Ascend310B1 bash scripts/step2_build_om.sh
```

SOC_VERSION 通过 `torch.npu.get_device_name()` 查询：`Ascend910_9382`、`Ascend310B1`。

### Step 3: ACL 推理评估

```bash
bash scripts/step3_eval_acl.sh
```

三后端通过 `--backend` 参数统一切换：

```bash
python scripts/eval_acl.py --config configs/cfg_udd5.py --backend pytorch
python scripts/eval_acl.py --config configs/cfg_udd5.py --backend onnx
python scripts/eval_acl.py --config configs/cfg_udd5.py --backend acl --om-dir models/om
```

### 310B1 推理

将 910 上编译的 OM 文件传到 310B1 设备，运行：

```bash
MAX_SAMPLES=5 bash scripts/run_310b1.sh    # 先试 5 张
bash scripts/run_310b1.sh                   # 全量
```

## 精度对比

UDD5 数据集全量 40 张评估结果，四后端精度完全对齐：

| 指标 | eval.py (MMSeg) | pytorch | onnx | acl (OM) |
|------|----------------|---------|------|----------|
| mIoU | 50.55 | 50.55 | 50.52 | 50.56 |
| aAcc | 74.86 | 74.85 | 74.84 | 74.87 |
| mAcc | 66.15 | 66.14 | 66.11 | 66.16 |

## 310B 算子适配

310B1 的 Cube 单元仅支持 FP16，且缺少多个 CUDA 常用算子。`npu_compat.py` 提供了等价替代：

| 原始算子 | 310B 问题 | 替代方案 |
|---------|----------|---------|
| `F.scaled_dot_product_attention` | 不支持 FlashAttention | bmm + softmax |
| `F.linear` (addmm) | MatMulV2 kernel 缺失 | matmul + add |
| `torch.linspace` | aclnnLinspace 未实现 | CPU 生成后拷贝 |
| `F.pad(reflect)` | aclnnReflectionPad2d 不可用 | index_select |
| `F.unfold` (im2col) | aclnnIm2col 不可用 | index_select + reshape |
| `F.interpolate(bilinear)` | 部分场景不可用 | 可分离 index_select |

> **注意：** `npu_compat.py` 仅用于 310B 上直接运行 PyTorch 推理的场景（`eval.py` / `demo.py`）。OM 离线推理不依赖它——相关算子在 ONNX 导出阶段（`scripts/export_onnx.py`）已通过等价图变换完成替换，编译后的 OM 模型由 ACL runtime 直接执行。

## NPU 项目结构

```
SegEarth-OV/
├── eval.py                     # PyTorch 评估 (已适配 NPU)
├── demo.py                     # 快速推理 demo (已适配 NPU)
├── npu_compat.py               # 310B 算子兼容层
├── custom_visualizer.py        # 三栏可视化
├── segearth_segmentor.py       # 模型定义 (cuda→npu)
├── ascend_310b_operators/      # 6 个 310B 自定义算子
├── scripts/
│   ├── export_onnx.py          # ONNX 导出 (310B 优化版)
│   ├── eval_acl.py             # 三后端统一评估
│   ├── step0_eval_pytorch.sh   # PyTorch 基线
│   ├── step1_export_onnx.sh    # ONNX 导出
│   ├── step2_build_om.sh       # ATC 编译
│   ├── step3_eval_acl.sh       # ACL 评估
│   └── run_310b1.sh            # 310B1 推理
├── models/
│   ├── onnx/                   # ONNX 模型产出
│   └── om/                     # OM 离线模型产出
├── logs/                       # 运行日志
├── utils/
│   ├── __init__.py
│   └── session.py              # PyTorch/ONNX/ACL Session 抽象
├── docs/
│   ├── deployment.md           # 部署流程文档
│   └── issues.md               # 问题记录
└── requirements_npu.txt        # NPU 依赖
```
