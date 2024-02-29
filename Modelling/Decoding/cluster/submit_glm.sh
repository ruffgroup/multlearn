#!/bin/bash
#SBATCH --job-name=fit_single_glm
#SBATCH --output=/home/ecasim/logs/fit_single_glm_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --time=30:00

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
module load anaconda3
source /home/${USER}/.bashrc
source activate multlearn-decoding

python $HOME/git/multlearn-sns/Modelling/Decoding/fit_glm.py $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/multlearn/ds-mlearn --data_folder /shares/zne.uzh/multlearn --rpe_folder /shares/zne.uzh/multlearn/bestFittingVals