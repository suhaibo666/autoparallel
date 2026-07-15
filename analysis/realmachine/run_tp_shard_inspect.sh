#!/usr/bin/env bash
set -eo pipefail

V4=/home/suhaibo/workspace/deepseek_v4/mindformers
R4=$V4/tests/st/test_multi_cards_cases/test_pynative/test_models/test_deepseekv4
HP=/home/suhaibo/workspace/deepseek_v4/hyper-parallel
PORT=${PORT:-8171}
CARDS=${CARDS:-6,7}
CFG=/tmp/codex_tp_shard_inspect.yaml
LOG=/tmp/codex_tp_shard_logs_${PORT}_$$

cp "$R4/dsv4_sim.yaml" "$CFG"
sed -i 's/tensor_parallel: 1/tensor_parallel: 2/' "$CFG"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/suhaibo/vendors/custom_transformer/bin/set_env.bash
set -u
export PATH=/root/miniconda3/envs/mindspore2.10/bin:$PATH
export PYTHONPATH=$V4:$HP:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES=$CARDS

mkdir -p "$LOG"
echo "[RUN_CONFIG] commit=$(git -C "$V4" rev-parse --short HEAD) cards=$CARDS port=$PORT log=$LOG"
/root/miniconda3/envs/mindspore2.10/bin/msrun \
  --worker_num=2 \
  --local_worker_num=2 \
  --master_port="$PORT" \
  --log_dir="$LOG" \
  --join=True \
  /tmp/codex_tp_shard_inspect.py \
  --config "$CFG"

grep -h '\[TPSHARD\]' "$LOG"/worker_*.log
echo "[LOGDIR] $LOG"
