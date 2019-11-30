#!/usr/bin/env bash
#SBATCH	-o evaluate
#SBATCH -p k80
#SBATCH --gres=gpu:1

conda deactivate
conda activate socialgan

cd Code
python scripts/evaluate_model.py --model_path ../checkpoints
