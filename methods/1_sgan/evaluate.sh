#!/usr/bin/env bash
#SBATCH	-o evaluate
#SBATCH -p k80
#SBATCH --gres=gpu:1

cd Code
echo $PWD > ~/.conda/envs/socialgan_with_torch1.3.1/lib/python3.5/site-packages/sgan.pth
python scripts/evaluate_model.py --model_path ../checkpoints
