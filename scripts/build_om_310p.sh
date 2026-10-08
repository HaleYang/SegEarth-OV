#!/usr/bin/env bash
# Compile the 310P ONNX exports into Ascend310P3 OM models.
#
# The PyTorch 2.12 exporter writes opset 18 even when opset 17 is requested.
# CANN 9.1.1's 310P parser accepts the graph after the standard-domain opset
# metadata is lowered to 17, so this script performs that local conversion
# before invoking ATC. It does not alter models/onnx_310p.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ONNX_DIR="${ONNX_DIR:-$REPO_ROOT/models/onnx_310p}"
OM_DIR="${OM_DIR:-$REPO_ROOT/models/om_310p}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/om_310p}"
SOC_VERSION="${SOC_VERSION:-Ascend310P3}"

mkdir -p "$OM_DIR" "$LOG_DIR"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/segearth_310p_onnx17.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

for name in clip_visual clip_text jbu_upsampler; do
    test -f "$ONNX_DIR/$name.onnx" || {
        echo "ERROR: missing $ONNX_DIR/$name.onnx" >&2
        exit 1
    }
done

python - "$ONNX_DIR" "$WORK_DIR" <<'PY'
from pathlib import Path
import shutil
import sys
import onnx

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
for name in ("clip_visual", "clip_text", "jbu_upsampler"):
    model = onnx.load(str(src / f"{name}.onnx"), load_external_data=False)
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset.version = 17
    onnx.save(model, str(dst / f"{name}.onnx"))
    data = src / f"{name}.onnx.data"
    if data.exists():
        shutil.copy2(data, dst / data.name)
PY

compile_model() {
    local name="$1"
    local shape="$2"
    local format="$3"
    echo "[ATC] $name -> $OM_DIR/$name.om"
    atc --model="$WORK_DIR/$name.onnx" \
        --framework=5 \
        --output="$OM_DIR/$name" \
        --soc_version="$SOC_VERSION" \
        --input_shape="$shape" \
        ${format:+--input_format="$format"} \
        --log=info 2>&1 | tee "$LOG_DIR/${name}.log"
}

compile_model clip_visual "image:1,3,224,224" NCHW
compile_model clip_text "text:8,77" ""
compile_model jbu_upsampler "source:1,512,14,14;guidance:1,3,224,224" NCHW

echo "Built OM models in $OM_DIR"
ls -lh "$OM_DIR"/*.om
