#!/usr/bin/env bash
#SBATCH	-o train_zara
#SBATCH -p k80
#SBATCH --gres=gpu:1
#SBATCH --time=4:25:00

echo $CONDA_DEFAULT_ENV
cd Code
echo $PWD > ~/.conda/envs/1_sgan/lib/python3.5/site-packages/sgan.pth
python scripts/train.py --output_dir ../checkpoints_zara --dataset_name zara1 --restore_from_checkpoint 1 --l2_loss_weight 1.0
