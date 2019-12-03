#!/usr/bin/env bash
#SBATCH	-o qual
#SBATCH -p k80
#SBATCH --gres=gpu:1
#SBATCH --time=4:25:00

echo $CONDA_DEFAULT_ENV
cd Code
echo $PWD > ~/.conda/envs/5_attention/lib/python3.5/site-packages/sgan.pth
python scripts/qualitative_eval.py --model_path ../checkpoints
