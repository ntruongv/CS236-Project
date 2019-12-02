#!/usr/bin/env bash
#SBATCH	-o qual
#SBATCH -p k80
#SBATCH --gres=gpu:1
#SBATCH --time=4:25:00

cd Code
echo $PWD > ~/.conda/envs/socialgan_with_torch1.3.1/lib/python3.5/site-packages/sgan.pth
python scripts/qualitative_eval.py --model_path ../checkpoints
