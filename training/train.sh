#!/bin/bash
# Launch the Long-WAM YAM conveyor fine-tune on the 2-GPU instance (run inside tmux).
#   bash training/train.sh smoke    # 30 optimizer steps: checks memory + measures s/step
#   bash training/train.sh action   # action expert only (~1B trainable), 4 epochs
#   bash training/train.sh full     # video + action experts (~6B), ZeRO-1 across both GPUs
# Extra Hydra overrides can follow, e.g. learning_rate=1e-4 num_epochs=6 batch_size=4
set -euo pipefail
MODE=${1:-smoke}; shift || true
ROOT=${LONGWAM_HOME:-$HOME/longwam}
REPO=$ROOT/LongLive/Long-WAM
cd $REPO
export PATH=$ROOT/.venv/bin:$PATH
export LONGWAM_CHECKPOINT_ROOT=$REPO/checkpoints LONGWAM_MODEL_ROOT=$REPO/models LONGWAM_DATA_ROOT=$REPO/data
export DIFFSYNTH_MODEL_BASE_PATH=$LONGWAM_MODEL_ROOT LONGWAM_OUTPUT_ROOT=$ROOT/outputs
export NPROC_PER_NODE=${NPROC_PER_NODE:-2} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Cloud VMs (e.g. Brev 2x RTX PRO 6000): the NCCL shared-memory transport fails with "illegal memory
# access" (low memlock limit); P2P works. Set NCCL_SHM_DISABLE=0 where SHM is fine.
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}
mkdir -p $LONGWAM_OUTPUT_ROOT/logs
case $MODE in
  smoke)  ARGS=(train_modules=action model.loss.lambda_video=0 max_steps=30 save_every=100000 save_final_weights=false save_final_state=false
                output_dir=$LONGWAM_OUTPUT_ROOT/smoke) ;;
  action) ARGS=(train_modules=action model.loss.lambda_video=0 learning_rate=1e-4) ;;
  full)   ARGS=(train_modules=all learning_rate=3e-5); export ACCELERATE_CONFIG=$REPO/configs/distributed/zero1.yaml ;;
  *) echo "mode must be smoke|action|full"; exit 2 ;;
esac
LOG=$LONGWAM_OUTPUT_ROOT/logs/${MODE}_$(date +%Y%m%d_%H%M%S).log
echo "logging to $LOG"
if [[ -n "${ACCELERATE_CONFIG:-}" ]]; then
  python -m accelerate.commands.launch --config_file "$ACCELERATE_CONFIG" --num_machines 1 --machine_rank 0 \
    --num_processes $NPROC_PER_NODE scripts/train.py task=yam_conveyor "${ARGS[@]}" "$@" 2>&1 | tee $LOG
else
  python -m torch.distributed.run --standalone --nproc_per_node=$NPROC_PER_NODE \
    scripts/train.py task=yam_conveyor "${ARGS[@]}" "$@" 2>&1 | tee $LOG
fi
