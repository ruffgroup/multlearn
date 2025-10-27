#!/bin/bash
#SBATCH --job-name=extract_rois
#SBATCH --output=/home/gdehol/logs/extract_rois_%A-%a.txt
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

python $HOME/git/multlearn-sns/SPM/code/extract_roi_timeseries.py \
    $PARTICIPANT_LABEL \
    --ROI $ROI