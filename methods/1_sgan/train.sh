#!/usr/bin/env bash
#SBATCH	-o train
#SBATCH -p k80
#SBATCH --gres=gpu:1
#SBATCH --time=4:25:00

cd Code
echo $PWD > ~/.conda/envs/socialgan_with_torch1.3.1/lib/python3.5/site-packages/sgan.pth
python scripts/train.py --output_dir ../checkpoints --dataset_name crowds_zara --restore_from_checkpoint 1 --l2_loss_weight 1.0
