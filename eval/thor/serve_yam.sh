#!/bin/bash
# Start / stop the Long-WAM-YAM policy server (TCP, default port 8780) on the Thor.
#   bash serve_yam.sh                       # foreground, pretrained checkpoint
#   bash serve_yam.sh --tmux                # in tmux session "lwserve"
#   bash serve_yam.sh --stop
#   CKPT_DIR=/path/to/finetuned bash serve_yam.sh --tmux   # model.pt + config.yaml + dataset_stats.json
# Env: LONGWAM_HOME (default ~/longwam), CKPT_DIR (default <repo>/checkpoints/yam), PORT (default 8780),
#      NUM_INFERENCE_STEPS (action denoising steps, default 10; the paper uses 4)
set -euo pipefail
ROOT=${LONGWAM_HOME:-$HOME/longwam}
REPO=$ROOT/LongLive/Long-WAM
CKPT_DIR=${CKPT_DIR:-$REPO/checkpoints/yam}
PORT=${PORT:-8780}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-10}
LOG=$ROOT/logs/serve_latest.log
case "${1:-}" in
  --stop) tmux kill-session -t lwserve 2>/dev/null && echo stopped || echo "not running"; exit 0 ;;
  --tmux) tmux kill-session -t lwserve 2>/dev/null || true
          tmux new-session -d -s lwserve "CKPT_DIR=$CKPT_DIR PORT=$PORT NUM_INFERENCE_STEPS=$NUM_INFERENCE_STEPS bash $0"
          echo "started tmux lwserve (log $LOG); ready when the log says 'listening on'"; exit 0 ;;
esac
cd $REPO
MODEL_CONFIG=$CKPT_DIR/config_infer.yaml
if [ ! -f $MODEL_CONFIG ]; then   # add the missing processor.train_transforms key once per checkpoint
  $ROOT/.venv/bin/python -c "
from omegaconf import OmegaConf
c = OmegaConf.load('$CKPT_DIR/config.yaml'); c.data.train.processor.train_transforms = None
OmegaConf.save(c, '$MODEL_CONFIG')"
fi
export LONGWAM_MODEL_ROOT=$REPO/models DIFFSYNTH_MODEL_BASE_PATH=$REPO/models HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p $ROOT/logs
exec $ROOT/.venv/bin/python $ROOT/yam_bridge/longwam_yam_server.py --ckpt $CKPT_DIR --port $PORT \
    --model_config $MODEL_CONFIG --num_inference_steps $NUM_INFERENCE_STEPS 2>&1 | tee $LOG
