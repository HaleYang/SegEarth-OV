"""Export the 310P JBU graph on CPU in an isolated process.

The CLIP visual/text graphs use NPU in ``export_onnx_310p.py``.  JBU is kept
in this separate process so torch_npu's asynchronous allocator state cannot
leak into a CPU ``.cpu()`` conversion.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(
    os.environ.get("SEG310P_OUTPUT_DIR", str(REPO_ROOT / "models" / "onnx_310p"))
).resolve()
BASELINE = REPO_ROOT / "scripts" / "export_onnx.py"


def _slice_patches(x, kernel_size, height, width):
    """Extract sliding windows without exporting F.unfold/Gather nodes."""
    return torch.stack([
        x[:, :, i:i + height, j:j + width]
        for i in range(kernel_size)
        for j in range(kernel_size)
    ], dim=2)


def _slice_weighted_sum(source, kernel, kernel_size, height, width):
    """Apply per-pixel kernels using Slice + Mul + Add only."""
    kernel_flat = kernel.reshape(kernel.shape[0], height, width,
                                 kernel_size * kernel_size).permute(0, 3, 1, 2)
    result = source[:, :, :height, :width] * kernel_flat[:, 0:1, :, :]
    for idx in range(1, kernel_size * kernel_size):
        i, j = divmod(idx, kernel_size)
        result = result + source[:, :, i:i + height, j:j + width] * \
            kernel_flat[:, idx:idx + 1, :, :]
    return result


def _load_baseline():
    spec = importlib.util.spec_from_file_location("segearth_onnx_baseline", BASELINE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load exporter: {BASELINE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jbu_forward_cpu(self, source, guidance):
    """JBU stage using the selected 310P export representation.

    The slice path is the default for 310P: it avoids ONNX Gather nodes from
    F.unfold and emits only static Slice/Mul/Add operations for ATC.
    """
    gb, _, gh, gw = guidance.shape
    _, sc, _, _ = source.shape
    k = self.diameter
    radius = self.radius

    dist_range = torch.linspace(-1, 1, k, device=source.device)
    x, y = torch.meshgrid(dist_range, dist_range, indexing="ij")
    patch = torch.stack((x, y), dim=0)
    spatial_kernel = torch.exp(
        -patch.square().sum(0) / (2 * self.sigma_spatial.float().square())
    ).reshape(1, k * k, 1, 1).to(source.dtype)

    proj_x = self.range_proj(guidance)
    proj_x_padded = F.pad(proj_x, pad=[radius] * 4, mode="reflect")
    use_slice = os.environ.get("SEG310P_JBU_UNFOLD_MODE", "slice") == "slice"
    if use_slice:
        patches = _slice_patches(proj_x_padded, k, gh, gw)
    else:
        patches = F.unfold(proj_x_padded, kernel_size=k).view(gb, -1, k * k, gh, gw)
    queries = patches.permute(0, 1, 3, 4, 2)
    pos_temp = self.range_temp.exp().clamp_min(1e-4).clamp_max(1e4)
    score = (queries * proj_x.unsqueeze(-1)).sum(dim=1)
    range_kernel = F.softmax(pos_temp * score, dim=-1).permute(0, 3, 1, 2)

    combined_kernel = range_kernel * spatial_kernel
    combined_kernel = combined_kernel / combined_kernel.sum(1, keepdim=True).clamp(1e-7)
    combined_kernel = combined_kernel + 0.1 * self.fixup_proj(
        torch.cat([combined_kernel, guidance], dim=1)
    )
    combined_kernel = combined_kernel.permute(0, 2, 3, 1).reshape(gb, gh, gw, k, k)

    # CANN 9.1.1 on 310P does not compile Resize(bicubic, half_pixel). The
    # baseline exporter already injects four depthwise-conv kernels that are
    # equivalent to bicubic 2x upsampling, so keep that representation here.
    ee = F.conv2d(
        F.pad(source, [2, 1, 2, 1], mode="replicate"), self._bk_ee, groups=sc
    )
    eo = F.conv2d(
        F.pad(source, [1, 2, 2, 1], mode="replicate"), self._bk_eo, groups=sc
    )
    oe = F.conv2d(
        F.pad(source, [2, 1, 1, 2], mode="replicate"), self._bk_oe, groups=sc
    )
    oo = F.conv2d(
        F.pad(source, [1, 2, 1, 2], mode="replicate"), self._bk_oo, groups=sc
    )
    even_row = torch.stack((ee, eo), dim=-1).reshape(gb, sc, source.shape[2], 2 * source.shape[3])
    odd_row = torch.stack((oe, oo), dim=-1).reshape(gb, sc, source.shape[2], 2 * source.shape[3])
    hr_source = torch.stack((even_row, odd_row), dim=-2).reshape(
        gb, sc, 2 * source.shape[2], 2 * source.shape[3]
    )
    hr_source = F.pad(hr_source, pad=[radius] * 4, mode="reflect")
    if use_slice:
        return _slice_weighted_sum(hr_source, combined_kernel, k, gh, gw)
    patches = F.unfold(hr_source, kernel_size=k).view(gb, sc, k * k, gh, gw)
    kernel_flat = combined_kernel.reshape(gb, gh, gw, k * k)
    return torch.einsum("bhwf,bcfhw->bchw", kernel_flat, patches)


def _pool_step(self, source, guidance, target_h, target_w):
    # Match JBUOne.upsample in the PyTorch implementation exactly.
    small_guidance = F.adaptive_avg_pool2d(guidance, (target_h, target_w))
    return self.up(source, small_guidance)


def main() -> None:
    exporter = _load_baseline()
    exporter.JBULearnedRange.forward = _jbu_forward_cpu
    exporter.JBUUpsamplerWrapper._upsample_step = _pool_step
    inject_bicubic_buffers = exporter._inject_bicubic_buffers

    def inject_cpu_buffers(module, C=512, dtype=torch.float16, device="cpu"):
        return inject_bicubic_buffers(module, C, dtype, device)

    exporter._inject_bicubic_buffers = inject_cpu_buffers

    feat_dim = 512
    # Export on CPU, but keep the FP16 interface expected by the ACL runtime.
    upsampler = exporter.get_upsampler("jbu_one", feat_dim).half().eval()
    ckpt_path = REPO_ROOT / "simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = {key[10:]: value for key, value in checkpoint["state_dict"].items()}
    upsampler.load_state_dict(weights, strict=True)
    wrapper = exporter.JBUUpsamplerWrapper(upsampler).half().eval()

    source = torch.randn(1, 512, 14, 14, dtype=torch.float16)
    guidance = torch.randn(1, 3, 224, 224, dtype=torch.float16)
    output_path = OUTPUT_DIR / "jbu_upsampler.onnx"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        output = wrapper(source, guidance)
        print(f"  input: source {source.shape}, guidance {guidance.shape}")
        print(f"  output: {output.shape}")
        torch.onnx.export(
            wrapper,
            (source, guidance),
            str(output_path),
            input_names=["source", "guidance"],
            output_names=["upsampled"],
            opset_version=17,
            dynamic_axes=None,
        )
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
