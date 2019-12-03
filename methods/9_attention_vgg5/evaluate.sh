#!/usr/bin/env bash
#SBATCH	-o evaluate
#SBATCH -p k80
#SBATCH --gres=gpu:1

echo $CONDA_DEFAULT_ENV
cd Code
python scripts/evaluate_model_w_local_context.py --model_path ../checkpoints
