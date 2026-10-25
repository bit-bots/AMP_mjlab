cd /homes/17vahl/smp/AMP_mjlab-kick-noballvel || exit 1
export PYTHONPATH=/homes/17vahl/smp/AMP_mjlab-kick-noballvel
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
RUN=$(ls -dt logs/rsl_rl/piplus_amp_kick/*/ | head -1)
CK=$(ls -t ${RUN}model_*.pt | head -1)
echo "checkpoint $CK"
exec timeout 1200 .venv/bin/python scripts/play.py PiPlus-AMP-Kick --checkpoint-file "$CK" --num-envs 1 --viewer viser
