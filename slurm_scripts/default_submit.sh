#!/usr/bin/env bash
#SBATCH	-o default_slurm_output
#SBATCH -p k80
#SBATCH --gres=gpu:1

cd CS236-Project/Code
python scripts/evaluate_model.py --model_path models/sgan-models/eth_8_model.pt
echo "!!!"
python scripts/evaluate_model.py --model_path ../../checkpoints_original/checkpoint_with_model.pt
