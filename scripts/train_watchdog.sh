#!/bin/bash
# train_watchdog.sh — supervise the RobotLab-Go2-v0 (MoE-CTS, IsaacLab) training:
#   * auto-resume from the newest checkpoint when the training process dies
#   * detect stalls (run train.log not moving for STALL_MIN minutes) and kill
#     the process so the resume path restarts it from the last checkpoint
#   * append a status line every ~30 minutes
#
# Usage:   train_watchdog.sh [interval_sec] [max_iterations] [num_envs] [task] [experiment_name]
#   defaults: interval=120s, max_iterations=50000, num_envs=16384,
#             task=RobotLab-Go2-v0, experiment_name=go2_moe_cts
# Stop:    kill $(cat /home/robot/go2_rl_robotlab/logs/train_watchdog.pid)
#          (or create /home/robot/go2_rl_robotlab/logs/train_watchdog.stop)
# Env:     WANDB_BASE_URL + WANDB_API_KEY must be exported before starting
#          (tmux sessions inherit the server env, so export explicitly)
#          WATCHDOG_DRY_RUN=1 (echo launches), WATCHDOG_STALL_MIN=30
set -u
INTERVAL="${1:-120}"
MAX_ITER="${2:-50000}"
NUM_ENVS="${3:-16384}"
TASK="${4:-RobotLab-Go2-v0}"
EXP="${5:-go2_moe_cts}"
BASE="$HOME/go2_rl_robotlab"
LOG_ROOT="$BASE/logs/rsl_rl/$EXP"
STOP_FILE="$BASE/logs/train_watchdog.stop"
PID_FILE="$BASE/logs/train_watchdog.pid"
DRY_RUN="${WATCHDOG_DRY_RUN:-0}"
STALL_MIN="${WATCHDOG_STALL_MIN:-30}"

echo "$$" > "$PID_FILE"
echo "[$(date '+%F %T')] watchdog started: interval=${INTERVAL}s max_iter=${MAX_ITER} num_envs=${NUM_ENVS} task=${TASK} exp=${EXP} pid=$$ stall_min=${STALL_MIN}"
loop=0
while true; do
  loop=$((loop + 1))
  if [ -f "$STOP_FILE" ]; then
    echo "[$(date '+%F %T')] stop file present; exiting"
    rm -f "$PID_FILE"
    exit 0
  fi
  if pgrep -f "train.py.*${TASK}" >/dev/null 2>&1; then
    # Stall detection: rsl_rl appends to the run's train.log every iteration.
    NEWEST_RUN=$(ls -td "$LOG_ROOT"/*/ 2>/dev/null | head -1)
    TRAIN_LOG="${NEWEST_RUN}train.log"
    LATEST=$(ls -t "$LOG_ROOT"/*/model_*.pt 2>/dev/null | head -1)
    AGE_MIN=""
    if [ -n "$NEWEST_RUN" ] && [ -f "$TRAIN_LOG" ]; then
      AGE_MIN=$((($(date +%s) - $(stat -c %Y "$TRAIN_LOG")) / 60))
      if [ "$AGE_MIN" -ge "$STALL_MIN" ]; then
        echo "[$(date '+%F %T')] STALL: ${TRAIN_LOG} has not moved for ${AGE_MIN} min; killing training to resume"
        pkill -f "train.py.*${TASK}" 2>/dev/null
        sleep 20
        pkill -9 -f "train.py.*${TASK}" 2>/dev/null
        sleep 5
      fi
    fi
    if [ $((loop % 15)) -eq 0 ]; then
      echo "[$(date '+%F %T')] status: training alive; log quiet ${AGE_MIN:-?} min; newest checkpoint ${LATEST##*/}; mem_avail=$(free -g | awk 'NR==2{print $7}')G"
    fi
    sleep "$INTERVAL"
    continue
  fi
  # Training is not running: resume from the newest checkpoint of the newest run.
  RUN_DIR=""
  CKPT=""
  for d in $(ls -td "$LOG_ROOT"/*/ 2>/dev/null); do
    c=$(ls -t "${d}"model_*.pt 2>/dev/null | head -1)
    if [ -n "$c" ]; then
      RUN_DIR="$d"
      CKPT="$c"
      break
    fi
  done
  if [ -z "$CKPT" ]; then
    echo "[$(date '+%F %T')] no checkpoint found under $LOG_ROOT; retrying"
    sleep "$INTERVAL"
    continue
  fi
  ITER=$(basename "$CKPT" | sed -E 's/model_([0-9]+)\.pt/\1/')
  if [ "$ITER" -ge "$MAX_ITER" ]; then
    echo "[$(date '+%F %T')] latest checkpoint ${ITER} >= ${MAX_ITER}; training complete; exiting"
    rm -f "$PID_FILE"
    exit 0
  fi
  RUN_NAME=$(basename "$RUN_DIR")
  STAMP=$(date '+%Y%m%d_%H%M%S')
  # NOTE: rsl_rl treats --max_iterations as ADDITIONAL iterations on resume, so
  # pass the remaining count to land exactly on MAX_ITER in total.
  REMAIN=$((MAX_ITER - ITER))
  LAUNCH="cd $BASE && .venv/bin/python scripts/rsl_rl/train.py --task=${TASK} --headless --num_envs ${NUM_ENVS} --max_iterations ${REMAIN} --logger wandb --resume --load_run ${RUN_NAME} --checkpoint $(basename "$CKPT") --run_name resume_${STAMP} 2>&1 | tee $BASE/logs/moects_resume_${STAMP}.log"
  echo "[$(date '+%F %T')] process gone; resuming from ${RUN_NAME}/$(basename "$CKPT") (${REMAIN} iters remaining to ${MAX_ITER})"
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] would launch: tmux new-session -d -s moects-r${STAMP} \"${LAUNCH}\""
  else
    tmux new-session -d -s "moects-r${STAMP}" "$LAUNCH"
  fi
  sleep 120
done
