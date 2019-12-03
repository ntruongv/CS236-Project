#!/usr/bin/env bash
#SBATCH	-o evaluate
#SBATCH -p k80
#SBATCH --gres=gpu:1

echo $CONDA_DEFAULT_ENV
cd Code
echo $PWD > ~/.conda/envs/2_hand_label/lib/python3.5/site-packages/sgan.pth
python scripts/evaluate_model_w_local_context.py --model_path ../checkpoints
