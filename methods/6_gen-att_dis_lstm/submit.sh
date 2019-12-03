#!/bin/bash

. /apps/software/anaconda3/5.2.0/etc/profile.d/conda.sh

conda deactivate
conda activate 6_gen-att_dis_lstm

sbatch $1
