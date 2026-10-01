cd /homes/17vahl/smp/AMP_mjlab-kick-noballvel || exit 1
export PYTHONPATH=/homes/17vahl/smp/AMP_mjlab-kick-noballvel
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
exec .venv/bin/python scripts/train.py PiPlus-AMP-Kick --env.scene.num-envs=4096
