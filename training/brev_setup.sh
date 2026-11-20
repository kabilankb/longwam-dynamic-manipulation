#!/bin/bash
# One-time setup of a GPU instance (tested: Brev 2x RTX PRO 6000 Blackwell) for the Long-WAM YAM fine-tune.
# Run ON the instance from a copy of this package (see README "Training on Brev"):
#   bash training/brev_setup.sh
# The converted dataset is taken from $ROOT/data/yam_conveyor/dataset if present (rsync it there),
# otherwise downloaded from $DATASET_REPO (default kabilanKB/yam_conveyor_cube_rand_longwam).
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_REPO=${DATASET_REPO:-kabilanKB/yam_conveyor_cube_rand_longwam}
ROOT=${LONGWAM_HOME:-$HOME/longwam}
REPO=$ROOT/LongLive/Long-WAM
mkdir -p $ROOT && cd $ROOT

# uv + Python 3.12 venv
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH UV_HTTP_TIMEOUT=600
[ -d LongLive ] || { git clone -q --filter=blob:none --sparse --single-branch --branch main --depth 1 \
    https://github.com/NVlabs/LongLive.git && (cd LongLive && git sparse-checkout set Long-WAM); }
cd $REPO
git checkout -q 31c70d9 2>/dev/null || true   # the commit the recipe was written against

# Local changes: YAM conveyor recipe + `train_modules: all|action` trainer option
cp "$PKG"/training/longwam_patches/configs/data/yam_conveyor.yaml configs/data/
cp "$PKG"/training/longwam_patches/configs/task/yam_conveyor.yaml configs/task/
(cd .. && git apply --check "$PKG"/training/longwam_patches/train_modules.patch 2>/dev/null \
   && git apply "$PKG"/training/longwam_patches/train_modules.patch) || grep -q train_modules src/longwam/trainer.py

[ -x $ROOT/.venv/bin/python ] || uv venv -q -p 3.12 $ROOT/.venv
export VIRTUAL_ENV=$ROOT/.venv
uv pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -q -e '.[train,test]'
# torchvision 0.22's VideoReader passes av.open(metadata_errors=...), which PyAV >= 15 removed.
uv pip install -q --only-binary av "av==14.2.0"
$ROOT/.venv/bin/python -c "import torch;print('torch',torch.__version__,'gpus',torch.cuda.device_count(),torch.cuda.get_device_name(0))"

# Weights: Long-WAM-YAM policy + Wan2.2 VAE + UMT5 text encoder/tokenizer
export LONGWAM_CHECKPOINT_ROOT=$REPO/checkpoints LONGWAM_MODEL_ROOT=$REPO/models
mkdir -p $LONGWAM_CHECKPOINT_ROOT $LONGWAM_MODEL_ROOT/Wan-AI
$ROOT/.venv/bin/hf download Efficient-Large-Model/Long-WAM-YAM \
    --revision e104e72dab1991a166e81e2941f7539e4edb8d68 --local-dir $LONGWAM_CHECKPOINT_ROOT/yam
(cd $LONGWAM_CHECKPOINT_ROOT/yam && sha256sum -c SHA256SUMS)
$ROOT/.venv/bin/hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth \
    --local-dir $LONGWAM_MODEL_ROOT/Wan-AI/Wan2.2-TI2V-5B
$ROOT/.venv/bin/hf download Wan-AI/Wan2.1-T2V-1.3B --include 'google/umt5-xxl/*' \
    --local-dir $LONGWAM_MODEL_ROOT/Wan-AI/Wan2.1-T2V-1.3B

# Data + text embeddings (one prompt). The converted dataset is public on HF, so pull it if not rsynced.
mkdir -p $ROOT/data/yam_conveyor
# (anonymous downloads of the 810 small files can hit HF rate limits: set HF_TOKEN, or rsync the dataset)
[ -f $ROOT/data/yam_conveyor/dataset/meta/info.json ] || $ROOT/.venv/bin/hf download \
    "$DATASET_REPO" --repo-type dataset --local-dir $ROOT/data/yam_conveyor/dataset
ln -sfn $ROOT/data $REPO/data
export LONGWAM_DATA_ROOT=$REPO/data DIFFSYNTH_MODEL_BASE_PATH=$LONGWAM_MODEL_ROOT
$ROOT/.venv/bin/python scripts/precompute_text_embeds.py task=yam_conveyor
$ROOT/.venv/bin/python scripts/precompute_text_embeds.py task=yam_conveyor +text_cache_mode=verify
echo "setup done; next: bash $PKG/training/train.sh [smoke|action|full]"
