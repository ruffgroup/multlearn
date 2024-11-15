#!/bin/bash
#SBATCH --job-name=fit_ridge_regr
#SBATCH --output=/home/ecasim/logs/fit_ridge_regr_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --time=30:00

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
module load anaconda3
source /home/${USER}/.bashrc
source activate multlearn-decoding

python $HOME/git/multlearn-sns/Modelling/Decoding/fit_ridge_regr.py $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/multlearn/ds-mlearn --data_folder /shares/zne.uzh/multlearn --mask 'v1'