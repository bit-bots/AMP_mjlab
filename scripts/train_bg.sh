#!/usr/bin/env bash
# Start scripts/train.py in the background without the per-iteration console
# summary in the log file (the same numbers are in tensorboard / wandb).
# Only stderr (warnings, errors, tracebacks) goes to logs/<name>.log.
#
#   scripts/train_bg.sh <name> PiPlus-AMP-BallStop --env.scene.num-envs=4096 ...
#
# Prints the PID of the training process.
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "usage: $0 <log name> <train.py args...>" >&2
  exit 1
fi
name="$1"
shift
cd "$(dirname "$0")/.."
mkdir -p logs
nohup .venv/bin/python scripts/train.py "$@" > /dev/null 2> "logs/${name}.log" &
echo $!
