#!/bin/bash
# launch_sgd_ablation.sh — wait for the running RobotLab-Go2-v0 training to
# finish, then run the SGD ablation experiment (single variable: optimizer only).
#
#   baseline : model_1500 --(Adam)--> 6500   [run 2026-09-15_01-34-30_resume_*]
#   ablation : model_1500 --(SGD) --> 6500   [this script]
#
# Everything else (environment, curriculum, hyper-parameters, target iteration)
# is identical; only the optimizer type differs.
#
# Stop:  kill $(cat ~/go2_rl_robotlab/logs/sgd_launcher.pid)
set -u
BASE="$HOME/go2_rl_robotlab"
LOG="$BASE/logs/sgd_ablation_launcher.log"
PIDFILE="$BASE/logs/sgd_launcher.pid"
SEED_DIR="$BASE/logs/rsl_rl/go2_moe_cts_sgd/seed_from_1500"
BASELINE_RUN="$BASE/logs/rsl_rl/go2_moe_cts/2026-09-14_22-25-39_moects_16k_50k_v1"
WANDB_ENV="WANDB_BASE_URL=http://127.0.0.1:8080 WANDB_API_KEY='local-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'"

echo "$$" > "$PIDFILE"
echo "[$(date '+%F %T')] launcher started; waiting for the current RobotLab-Go2-v0 run to finish" >> "$LOG"
while pgrep -f "train.py.*RobotLab-Go2-v0" >/dev/null 2>&1; do sleep 120; done
echo "[$(date '+%F %T')] current run finished; waiting 60s for the GPU to free" >> "$LOG"
sleep 60

# Seed the ablation experiment dir with the exact fork checkpoint so that even a
# pre-first-save crash resumes the *correct* experiment (watchdog safety).
mkdir -p "$SEED_DIR"
cp -n "$BASELINE_RUN/model_1500.pt" "$SEED_DIR/model_1500.pt"
echo "[$(date '+%F %T')] seeded $SEED_DIR/model_1500.pt" >> "$LOG"

# 1) smoke test the SGD task (tiny run on the free GPU)
cd "$BASE" || exit 1
timeout 1800 .venv/bin/python scripts/rsl_rl/train.py --task=RobotLab-Go2-MoECTSSGD-v0 --headless --num_envs 64 --max_iterations 2 --run_name sgd_smoke >> "$LOG" 2>&1
SMOKE=$?
if [ $SMOKE -ne 0 ]; then
  echo "[$(date '+%F %T')] SMOKE FAILED (exit $SMOKE); aborting; see $LOG" >> "$LOG"
  exit 1
fi
echo "[$(date '+%F %T')] smoke ok; launching the full SGD ablation" >> "$LOG"
rm -rf "$BASE"/logs/rsl_rl/go2_moe_cts_sgd/*sgd_smoke* 2>/dev/null

# 2) full ablation: seed_from_1500 -> 6500 (5000 additional iterations)
tmux new-session -d -s moects-sgd "cd $BASE && $WANDB_ENV .venv/bin/python scripts/rsl_rl/train.py --task=RobotLab-Go2-MoECTSSGD-v0 --headless --num_envs 16384 --max_iterations 5000 --logger wandb --resume --load_run seed_from_1500 --checkpoint model_1500.pt --run_name sgd_from_1500_to_6500 2>&1 | tee $BASE/logs/moects_sgd_from_1500.log"
sleep 90

# 3) watchdog for the ablation (target 6500, experiment dir go2_moe_cts_sgd)
tmux new-session -d -s watchdog-sgd "cd $BASE && $WANDB_ENV bash scripts/train_watchdog.sh 120 6500 16384 RobotLab-Go2-MoECTSSGD-v0 go2_moe_cts_sgd"
echo "[$(date '+%F %T')] SGD ablation + watchdog launched" >> "$LOG"
