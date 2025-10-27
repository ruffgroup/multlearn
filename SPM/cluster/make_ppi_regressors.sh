#!/bin/bash
#SBATCH --job-name=make_ppi_regressors
#SBATCH --output=/home/gdehol/logs/make_ppi_regressors_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --time=15:00

. $HOME/init_conda.sh

module load matlab
module unload python
conda activate multlearn

# Zero-pad the SLURM_ARRAY_TASK_ID to two digits
PARTICIPANT_LABEL=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})

# Capture the ROI argument
ROI=$1
VARIABLE=$2

python $HOME/git/multlearn-sns/SPM/code/make_ppi_regressors.py \
    $PARTICIPANT_LABEL \
    --roi $ROI \
    --variable $VARIABLE