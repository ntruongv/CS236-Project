#!/usr/bin/env bash
#SBATCH	-o evaluate_600
#SBATCH -p k80
#SBATCH --gres=gpu:1

echo $CONDA_DEFAULT_ENV
cd Code
echo $PWD > ~/.conda/envs/5_attention/lib/python3.5/site-packages/sgan.pth
python scripts/evaluate_model_w_local_context.py --model_path ../checkpoints_600
