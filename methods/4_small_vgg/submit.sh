#!/bin/bash

. /apps/software/anaconda3/5.2.0/etc/profile.d/conda.sh

conda deactivate
conda activate 4_small_vgg

sbatch $1
