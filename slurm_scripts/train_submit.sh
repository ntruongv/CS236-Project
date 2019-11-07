#!/usr/bin/env bash
#SBATCH	-o train_slurm_output
#SBATCH -p k80
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

cd CS236-Project/Code
python scripts/train_w_local_context.py --output_dir ../../checkpoints --dataset_name crowds_zara --restore_from_checkpoint 1 --l2_loss_weight 1.0
