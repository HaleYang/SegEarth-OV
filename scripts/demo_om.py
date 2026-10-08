#!/usr/bin/env python3
"""SegEarth-OV OM 推理演示脚本。

轻量级单图 / 批量推理 + 可视化，不依赖 MMSeg Runner。

用法:
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

    # 单张推理 + 可视化
    python scripts/demo_om.py -i data/UDD/UDD/UDD5/val/src/000061.JPG

    # 批量推理整个目录
    python scripts/demo_om.py -i data/UDD/UDD/UDD5/val/src/ -o demo_output/

    # 指定配置 / OM 目录
    python scripts/demo_om.py -i img.jpg --config configs/cfg_udd5.py \
        --om-dir models/om
"""
import os, sys, time, argparse
import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import custom_datasets  # noqa: F401
from mmengine.config import Config
from mmseg.registry import DATASETS


# ---------------------------------------------------------------------------
#  轻量可视化 (纯 PIL, 不依赖 mmcv / MMSeg visualizer)
# ---------------------------------------------------------------------------

def colorize_mask(mask, palette, alpha=0.55):
    """将分割 mask 转为 RGBA 彩色图。"""
    h, w = mask.shape
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    for cls_id, color in enumerate(palette):
        m = mask == cls_id
        canvas[m, :3] = color
        canvas[m, 3] = int(alpha * 255)
    return Image.fromarray(canvas, 'RGBA')


def _draw_legend(palette, class_names, item_h=22, font_size=14):
    """绘制竖排图例条, 返回 PIL Image (RGB)。"""
    from PIL import ImageDraw, ImageFont
    n = len(class_names)
    swatch = item_h - 4
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()
    max_tw = max(font.getlength(c) for c in class_names)
    legend_w = swatch + 8 + int(max_tw) + 12
    legend_h = item_h * n + 8
    img = Image.new('RGB', (legend_w, legend_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i, (name, color) in enumerate(zip(class_names, palette)):
        y = 4 + i * item_h
        draw.rectangle([4, y, 4 + swatch, y + swatch], fill=tuple(color))
        draw.text((4 + swatch + 6, y), name, fill=(0, 0, 0), font=font)
    return img


def make_vis(img_pil, seg_pred, palette, class_names, max_side=960):
    """生成 [原图 | 叠加预测 | 图例] 可视化, 限制最大边长。"""
    w, h = img_pil.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img_small = img_pil.resize((new_w, new_h), Image.BILINEAR)
        pred_small = np.array(Image.fromarray(seg_pred.astype(np.uint8)).resize(
            (new_w, new_h), Image.NEAREST))
    else:
        img_small = img_pil
        pred_small = seg_pred
        new_w, new_h = w, h

    overlay = colorize_mask(pred_small, palette)
    blended = Image.alpha_composite(img_small.convert('RGBA'), overlay).convert('RGB')

    legend = _draw_legend(palette, class_names)
    lw, lh = legend.size

    gap = 4
    total_w = new_w * 2 + gap + lw + gap
    canvas = Image.new('RGB', (total_w, max(new_h, lh)), (255, 255, 255))
    canvas.paste(img_small.convert('RGB'), (0, 0))
    canvas.paste(blended, (new_w + gap, 0))
    canvas.paste(legend, (new_w * 2 + gap * 2, 0))
    return canvas


# ---------------------------------------------------------------------------
#  预处理 (与 eval_acl.py preprocess_numpy 一致)
# ---------------------------------------------------------------------------

def _keep_ratio_size(orig_w, orig_h, target_size):
    scale = min(target_size / orig_h, target_size / orig_w)
    return int(orig_w * scale + 0.5), int(orig_h * scale + 0.5)


def preprocess(img_pil, size):
    orig_w, orig_h = img_pil.size
    new_w, new_h = _keep_ratio_size(orig_w, orig_h, size)
    arr = np.array(img_pil, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    resized = np.stack([
        np.array(Image.fromarray(arr[c], mode="F").resize(
            (new_w, new_h), Image.BILINEAR))
        for c in range(3)
    ])
    return resized[np.newaxis, ...].astype(np.float16)


# ---------------------------------------------------------------------------
#  主流程
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='SegEarth-OV OM demo')
    p.add_argument('-i', '--input', required=True,
                   help='输入图片路径或目录')
    p.add_argument('-o', '--output', default='demo_output',
                   help='输出目录 (默认 demo_output/)')
    p.add_argument('--config', default='configs/cfg_udd5.py')
    p.add_argument('--om-dir', default='models/om')
    p.add_argument('--device-id', type=int, default=0)
    p.add_argument('--size', type=int, default=448)
    p.add_argument('--template', choices=['full', 'sub'], default='full')
    p.add_argument('--prob-thd', type=float, default=None,
                   help='覆盖配置中的预测置信度阈值')
    p.add_argument('--max-side', type=int, default=960,
                   help='可视化最大边长 (越小越快)')
    p.add_argument('--no-vis', action='store_true',
                   help='只输出 mask, 不生成可视化')
    return p.parse_args()


def collect_images(path):
    exts = {'.jpg', '.jpeg', '.png', '.tif', '.bmp'}
    if os.path.isfile(path):
        return [path]
    return sorted([os.path.join(path, f) for f in os.listdir(path)
                   if os.path.splitext(f)[1].lower() in exts])


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    from segearth_segmentor import get_cls_idx
    name_path = cfg.model.get('name_path', '')
    query_words, query_idx = get_cls_idx(name_path)
    num_classes = max(query_idx) + 1

    name_list = [l.strip() for l in open(name_path) if l.strip()]

    ds_type = cfg.test_dataloader.dataset.type
    ds_cls = DATASETS.module_dict.get(ds_type)
    palette = ds_cls.METAINFO.get('palette', None) if ds_cls else None
    if palette is None:
        np.random.seed(42)
        palette = np.random.randint(0, 255, (num_classes, 3)).tolist()

    from utils.session import Session, SessionConfig
    session_cfg = SessionConfig(
        session_type='acl',
        om_dir=args.om_dir,
        device_id=args.device_id,
        name_list=name_list,
        prob_thd=(args.prob_thd if args.prob_thd is not None
                  else cfg.model.get('prob_thd', 0.1)),
        logit_scale=cfg.model.get('logit_scale', 50.0),
        bg_idx=cfg.model.get('bg_idx', 0),
        cls_token_lambda=cfg.model.get('cls_token_lambda', -0.3),
        template=args.template,
    )

    print(f"Loading OM models from {args.om_dir} ...")
    session = Session.from_config(session_cfg)

    images = collect_images(args.input)
    print(f"Found {len(images)} image(s)")

    os.makedirs(args.output, exist_ok=True)

    total_infer = 0.0
    for idx, img_path in enumerate(images):
        img_pil = Image.open(img_path).convert('RGB')
        ori_w, ori_h = img_pil.size
        img_input = preprocess(img_pil, args.size)

        t0 = time.time()
        seg_pred = session.predict(img_input, ori_shape=(ori_h, ori_w))
        t_infer = time.time() - t0
        total_infer += t_infer

        if seg_pred.shape != (ori_h, ori_w):
            seg_pred = np.array(Image.fromarray(seg_pred.astype(np.uint8)).resize(
                (ori_w, ori_h), Image.NEAREST))

        base = os.path.splitext(os.path.basename(img_path))[0]

        mask_path = os.path.join(args.output, f'{base}_mask.png')
        Image.fromarray(seg_pred.astype(np.uint8)).save(mask_path)
        # Keep the indexed mask for evaluation, and also emit a human-readable
        # palette image. Background is intentionally black in the palette.
        palette_arr = np.asarray(palette, dtype=np.uint8)
        color_mask = palette_arr[np.clip(seg_pred.astype(np.int64), 0,
                                         len(palette_arr) - 1)]
        color_mask_path = os.path.join(args.output, f'{base}_mask_color.png')
        Image.fromarray(color_mask, 'RGB').save(color_mask_path)

        if not args.no_vis:
            t1 = time.time()
            vis = make_vis(img_pil, seg_pred, palette, name_list, args.max_side)
            vis_path = os.path.join(args.output, f'{base}_vis.jpg')
            vis.save(vis_path, quality=90)
            t_vis = time.time() - t1
        else:
            t_vis = 0

        print(f"  [{idx+1}/{len(images)}] {base}  "
              f"infer={t_infer:.2f}s  vis={t_vis:.2f}s  "
              f"pred_classes={np.unique(seg_pred).tolist()}")

    session.close()

    avg = total_infer / len(images) if images else 0
    print(f"\nDone. {len(images)} images, avg infer {avg:.2f}s/img")
    print(f"Results in {args.output}/")


if __name__ == "__main__":
    main()
