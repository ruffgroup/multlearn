#!/bin/bash
#SBATCH --job-name=fit_single_glm
#SBATCH --output=/home/ecasim/logs/fit_pupil_glm_%A-%a.txt
#SBATCH --partition=generic
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --time=60:00

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

python $HOME/git/multlearn-sns/Modelling/Decoding/fit_pupil_glm.py $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/multlearn/ds-mlearn --data_folder /shares/zne.uzh/multlearn --rpe_folder /shares/zne.uzh/multlearn/multlearn/bestFittingVals