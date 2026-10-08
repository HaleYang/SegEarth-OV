"""Export the SegEarth-OV ONNX graphs with the 310P native-op candidates.

This is intentionally separate from ``export_onnx.py`` (the 310B baseline).
The baseline remains the compatibility reference.  For the first 310P pass we
restore the native PyTorch graph for argmax, unfold, and interpolation while
keeping the explicit attention graph used by the baseline export.

Usage::

    python scripts/export_onnx_310p.py

The generated files are written to ``models/onnx_310p``.  Set
``SEG310P_OUTPUT_DIR`` to override that directory.
"""

from __future__ import annotations

import os
import re
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "scripts" / "export_onnx.py"
OUTPUT_DIR = Path(
    os.environ.get("SEG310P_OUTPUT_DIR", str(REPO_ROOT / "models" / "onnx_310p"))
).resolve()


def main() -> None:
    """Run the 310B exporter with the 310P native-op substitutions."""
    source = SOURCE.read_text(encoding="utf-8")

    # Keep the CLIP exports entirely on NPU. JBU is exported by the separate
    # CPU script below; moving a live NPU module to CPU in this process can
    # trigger torch_npu stream/allocator errors.
    source = source.replace(
        "    export_upsampler_combined()\n    export_upsampler_stages()\n", ""
    )

    # The current torch_npu MHA path fails before ONNX tracing on 310P
    # (format-cast error). Inject an explicit attention implementation into
    # every CLIP ResidualAttentionBlock used by visual and text encoders.
    attention_patch = r'''

def _onnx_safe_attention(self, q_x, k_x=None, v_x=None, attn_mask=None):
    k_x = q_x if k_x is None else k_x
    v_x = q_x if v_x is None else v_x
    length, batch, embed_dim = q_x.shape
    heads = self.attn.num_heads
    head_dim = embed_dim // heads
    qkv = torch.matmul(q_x, self.attn.in_proj_weight.transpose(0, 1))
    if self.attn.in_proj_bias is not None:
        qkv = qkv + self.attn.in_proj_bias
    q = qkv[..., :embed_dim]
    k = qkv[..., embed_dim:2 * embed_dim]
    v = qkv[..., 2 * embed_dim:]
    # Match torch.nn.MultiheadAttention's native layout exactly: flatten the
    # batch/head dimensions after splitting the embedding dimension, rather
    # than permuting to [B, H, L, D] (which changes token/head ordering).
    q = q.contiguous().view(length, batch * heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(length, batch * heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(length, batch * heads, head_dim).transpose(0, 1)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
    if attn_mask is not None:
        # Scores use the native flattened [B*H, L, L] layout, so the causal
        # mask must remain [L, L]. Adding [1, 1, L, L] would broadcast an
        # unintended fourth dimension and corrupt text features.
        scores = scores + attn_mask.to(scores.dtype).reshape(length, length)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v).transpose(0, 1).contiguous().view(length, batch, embed_dim)
    out = torch.matmul(out, self.attn.out_proj.weight.transpose(0, 1))
    if self.attn.out_proj.bias is not None:
        out = out + self.attn.out_proj.bias
    return out

from open_clip.transformer import ResidualAttentionBlock
ResidualAttentionBlock.attention = _onnx_safe_attention
'''
    marker = "from open_clip import create_model, tokenizer"
    source = source.replace(marker, marker + attention_patch, 1)

    # Keep this transformation small and explicit so the 310B exporter remains
    # the reference implementation and every 310P graph change is auditable.
    source = re.sub(
        r"    @staticmethod\n    def _argmax_no_argmax\(x\):.*?        return masked\.max\(dim=-1\)\.values\.long\(\)\n",
        "    @staticmethod\n    def _argmax_no_argmax(x):\n        return x.argmax(dim=-1)\n",
        source,
        count=1,
        flags=re.S,
    )

    # The 310B exporter sets its output directory as a module constant.  Keep
    # the generated graphs separate from existing 310B ONNX artifacts.
    source = source.replace(
        'OUTPUT_DIR = os.path.join(REPO_ROOT, "models", "onnx")',
        f'OUTPUT_DIR = r"{OUTPUT_DIR}"',
    )
    source = source.replace(
        'REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))',
        f'REPO_ROOT = r"{REPO_ROOT}"',
    )
    source = source.replace(
        'print("\\nDone! All ONNX files exported to models/onnx/")',
        f'print("\\nDone! All ONNX files exported to {OUTPUT_DIR}/")',
    )

    # Keep the baseline depthwise-conv bicubic implementation: 310P CANN
    # rejects Resize(bicubic, half_pixel) during ATC compilation.
    source = source.replace(
        "result = _adaptive_conv_patched(hr_source_padded, combined_kernel)",
        "patches = F.unfold(hr_source_padded, kernel_size=K).view(GB, SC, K * K, GH, GW)\n    kernel_flat = combined_kernel.reshape(GB, GH, GW, K * K)\n    result = torch.einsum('bhwf,bcfhw->bchw', kernel_flat, patches)",
    )
    source = source.replace(
        "F.adaptive_avg_pool2d(guidance, (target_h, target_w))",
        "F.interpolate(guidance, (target_h, target_w), mode='bilinear', align_corners=False)",
    )
    # The ACL path pads text batches to 8 before inference. Keep this graph
    # static; dynamic batch shape tensors make ATC reshape inference fail.
    source = source.replace(
        'dynamic_axes={"text": {0: "batch"}, "text_features": {0: "batch"}}',
        'dynamic_axes=None',
    )
    source = source.replace(
        'tokenizer.tokenize(["a photo of a building"])',
        'tokenizer.tokenize(["a photo of a building"] * 8)',
    )
    # CANN 9.1.1 on 310P rejects the three-output Split emitted by
    # ``torch.chunk(3)``. Explicit slices preserve the same Q/K/V values.
    source = re.sub(
        r"(?m)^([ \t]*)q, k, v = qkv\.chunk\(3, dim=-1\)$",
        lambda match: (
            f"{match.group(1)}q = qkv[..., :embed_dim]\n"
            f"{match.group(1)}k = qkv[..., embed_dim:2 * embed_dim]\n"
            f"{match.group(1)}v = qkv[..., 2 * embed_dim:]"
        ),
        source,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_export_onnx_310p.py", encoding="utf-8", delete=False
    ) as handle:
        handle.write(source)
        generated = Path(handle.name)
    try:
        runpy.run_path(str(generated), run_name="__main__")
    finally:
        generated.unlink(missing_ok=True)

    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "export_jbu_onnx_310p_cpu.py")],
        check=True,
        env=os.environ.copy(),
    )


if __name__ == "__main__":
    main()
