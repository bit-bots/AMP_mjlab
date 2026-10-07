cd /homes/17vahl/smp/AMP_mjlab-kick-noballvel || exit 1
export PYTHONPATH=/homes/17vahl/smp/AMP_mjlab-kick-noballvel
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
CK=logs/rsl_rl/piplus_amp_kick/2026-09-30_20-22-31/model_34000.pt
echo "checkpoint $CK"
exec timeout 1200 .venv/bin/python scripts/play.py PiPlus-AMP-Kick --checkpoint-file "$CK" --num-envs 1 --viewer viser
