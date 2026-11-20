#!/bin/bash
# Generate a LongLive2.0-Robot-S "imagination" video (start image + instruction -> 5 s, 1280x704, 24 fps)
# on a Jetson Thor that already ran eval/thor/setup_thor.sh. Run from the workstation:
#   bash video/run_on_thor.sh USER@THOR_HOST [IMAGE.png] ["prompt"]
# FlashAttention is not required: video/stage1_sdpa_fallback.patch makes Stage 1 fall back to PyTorch SDPA.
set -euo pipefail
THOR=${1:?usage: run_on_thor.sh USER@THOR_HOST [image.png] ["prompt"]}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=${2:-$HERE/conveyor_start_1280x704.png}
PROMPT=${3:-right gripper picks up the red cube from the conveyor belt and places it into the blue bin}
scp -q "$HERE/stage1_sdpa_fallback.patch" "$IMAGE" "$THOR":/tmp/
ssh "$THOR" IMAGE="/tmp/$(basename "$IMAGE")" PROMPT="'$PROMPT'" 'bash -s' <<'REMOTE'
set -euo pipefail
ROOT=${LONGWAM_HOME:-$HOME/longwam}; REPO=$ROOT/LongLive/Long-WAM
export PATH=$HOME/.local/bin:$PATH VIRTUAL_ENV=$ROOT/.venv UV_HTTP_TIMEOUT=600
cd $ROOT/LongLive
git apply --check /tmp/stage1_sdpa_fallback.patch 2>/dev/null && git apply /tmp/stage1_sdpa_fallback.patch || true
cd $REPO
# Stage-1 extras (PyAV from the Long-WAM env is kept: building PyAV 13/14 from source fails on JetPack's FFmpeg)
uv pip install -q "diffusers==0.31.0" easydict omegaconf einops ftfy imageio imageio-ffmpeg sentencepiece peft
mkdir -p checkpoints/robot-s
[ -s checkpoints/robot-s/model.pt ] || hf download Efficient-Large-Model/LongLive2.0-Robot-S \
    --revision 9f23acf1a619afe69d866f105c17209880851894 --local-dir checkpoints/robot-s
(cd checkpoints/robot-s && sha256sum -c SHA256SUMS)
A=$ROOT/video_assets/wan_models/Wan2.2-TI2V-5B; mkdir -p $A/google
ln -sfn $REPO/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth $A/Wan2.2_VAE.pth
ln -sfn $REPO/models/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth $A/models_t5_umt5-xxl-enc-bf16.pth
ln -sfn $REPO/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl $A/google/umt5-xxl
[ -f $A/config.json ] || hf download Wan-AI/Wan2.2-TI2V-5B config.json --local-dir $A
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas   # the venv's Triton ptxas lacks sm_110a
OUT=$ROOT/video_gen/out_$(date +%Y%m%d_%H%M%S); mkdir -p $OUT
tmux kill-session -t lwvideo 2>/dev/null || true
tmux new-session -d -s lwvideo "cd $REPO && PYTHON=$ROOT/.venv/bin/python bash stage1/infer.sh --image $IMAGE \
  --prompt $PROMPT --checkpoint checkpoints/robot-s/model.pt --assets-root $ROOT/video_assets \
  --output-dir $OUT 2>&1 | tee $OUT/infer.log"
echo "generating in tmux lwvideo -> $OUT/custom.mp4 (about 9 min on Thor)"
REMOTE
