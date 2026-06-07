#!/bin/bash

# activate conda
source ~/miniconda3/etc/profile.d/conda.sh
# source ~/.bashrc
conda activate robot

# run user command
exec "$@"
