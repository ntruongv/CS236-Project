#!/usr/bin/env bash
#SBATCH	-o train_original_slurm_output
#SBATCH -p k80
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

cd CS236-Project/Code
python scripts/train.py --output_dir ../../checkpoints_original --dataset_name crowds_zara --restore_from_checkpoint 1 --l2_loss_weight 1.0
