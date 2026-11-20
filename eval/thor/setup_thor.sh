#!/bin/bash
# One-time setup of the Long-WAM-YAM policy server on a Jetson AGX Thor (JetPack 7, CUDA 13, aarch64).
# Run ON the Thor from a copy of this package:   bash eval/thor/setup_thor.sh
# Afterwards start the server with eval/thor/serve_yam.sh (see the README).
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT=${LONGWAM_HOME:-$HOME/longwam}
REPO=$ROOT/LongLive/Long-WAM
export PATH=$HOME/.local/bin:$PATH UV_HTTP_TIMEOUT=600
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p $ROOT && cd $ROOT
[ -d LongLive ] || { git clone -q --filter=blob:none --sparse --single-branch --branch main --depth 1 \
    https://github.com/NVlabs/LongLive.git && (cd LongLive && git sparse-checkout set Long-WAM); }

[ -x $ROOT/.venv/bin/python ] || uv venv -q -p 3.12 $ROOT/.venv
export VIRTUAL_ENV=$ROOT/.venv
# Thor needs a CUDA 13 aarch64 build (sm_110); Long-WAM's own pin (<2.8) has none, so install it --no-deps.
uv pip install -q torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install -q "accelerate==1.12.0" "hydra-core==1.3.2" "omegaconf==2.3.0" "einops==0.8.1" "numpy>=2.2.6,<2.3" \
    pillow "safetensors>=0.5" "transformers==4.49.0" "huggingface-hub>=0.29,<1" sentencepiece ftfy regex tqdm rich \
    termcolor "imageio>=2.37" "imageio-ffmpeg>=0.6" "PyYAML>=6" "opencv-python-headless>=4.9,<4.12" \
    "datasets>=4,<5" pandas pyarrow jsonlines ninja packaging psutil "setuptools>=77.0.3"
(cd $REPO && uv pip install -q -e . --no-deps)
$ROOT/.venv/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

# Policy checkpoint + Wan2.2 VAE + UMT5 text encoder/tokenizer
cd $REPO
mkdir -p checkpoints models/Wan-AI
[ -f checkpoints/yam/model.pt ] || hf download Efficient-Large-Model/Long-WAM-YAM \
    --revision e104e72dab1991a166e81e2941f7539e4edb8d68 --local-dir checkpoints/yam
(cd checkpoints/yam && sha256sum -c SHA256SUMS)
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth --local-dir models/Wan-AI/Wan2.2-TI2V-5B
hf download Wan-AI/Wan2.1-T2V-1.3B --include 'google/umt5-xxl/*' --local-dir models/Wan-AI/Wan2.1-T2V-1.3B

# The released YAM config.yaml lacks processor.train_transforms (needed to build the processor).
$ROOT/.venv/bin/python - <<'EOF'
from omegaconf import OmegaConf
c = OmegaConf.load("checkpoints/yam/config.yaml")
c.data.train.processor.train_transforms = None
OmegaConf.save(c, "checkpoints/yam/config_infer.yaml")
EOF

mkdir -p $ROOT/yam_bridge $ROOT/logs
cp "$PKG"/eval/thor/{longwam_yam_server.py,wire.py,mock_server.py,serve_yam.sh} $ROOT/yam_bridge/
echo "Thor setup done. Start the server: bash $ROOT/yam_bridge/serve_yam.sh --tmux"
