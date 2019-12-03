#!/usr/bin/env bash
#SBATCH	-o repeat
#SBATCH --time=4:30:00

cd /home/dansj/CS236-Project/methods/5_attention
./submit.sh train_zara.sh
sleep 16100
sbatch repeat_train_zara.sh
