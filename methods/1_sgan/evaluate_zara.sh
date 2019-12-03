#!/usr/bin/env bash
#SBATCH	-o evaluate_zara
#SBATCH -p k80
#SBATCH --gres=gpu:1

echo $CONDA_DEFAULT_ENV
cd Code
echo $PWD > ~/.conda/envs/1_sgan/lib/python3.5/site-packages/sgan.pth
python scripts/evaluate_model.py --model_path ../checkpoints_zara
