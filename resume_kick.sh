cd /homes/17vahl/smp/AMP_mjlab-kick-noballvel || exit 1
export PYTHONPATH=/homes/17vahl/smp/AMP_mjlab-kick-noballvel
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
exec .venv/bin/python scripts/train.py PiPlus-AMP-Kick --env.scene.num-envs=4096 --agent.resume True --agent.load-run 2026-10-02_22-33-48 --agent.load-checkpoint model_52000.pt
