#!/usr/bin/env bash
#SBATCH	-o col
#SBATCH -p k80
#SBATCH --gres=gpu:1
#SBATCH --time=4:25:00

echo $CONDA_DEFAULT_ENV
cd Code
python scripts/collisions_eval.py --model_path ../checkpoints
