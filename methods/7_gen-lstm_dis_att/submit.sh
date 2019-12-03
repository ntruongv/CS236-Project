#!/bin/bash

. /apps/software/anaconda3/5.2.0/etc/profile.d/conda.sh

conda deactivate
conda activate 7_gen-lstm_dis_att

sbatch $1
