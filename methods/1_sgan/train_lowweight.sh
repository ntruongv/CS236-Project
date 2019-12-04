#!/usr/bin/env bash
#SBATCH	-o train_lowweight
#SBATCH -p k80
#SBATCH --gres=gpu:1
#SBATCH --time=4:25:00

echo $CONDA_DEFAULT_ENV
cd Code
echo $PWD > ~/.conda/envs/1_sgan/lib/python3.5/site-packages/sgan.pth
python scripts/train.py --output_dir ../checkpoints_lowweight --dataset_name crowds_zara --restore_from_checkpoint 1 --l2_loss_weight 0.01 --num_epochs 400 --batch_size 32
