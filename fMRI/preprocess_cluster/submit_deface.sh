#!/bin/bash
#SBATCH --job-name=deface
#SBATCH --output=/home/%u/logs/deface_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# SLURM script to deface anatomical images
# Usage: sbatch submit_deface.sh
# Or for a specific subject: sbatch --array=1 submit_deface.sh

# Initialize conda (adjust path if needed)
. $HOME/init_conda.sh

conda activate multlearn

# BIDS folder path
BIDS_FOLDER="/shares/zne.uzh/multlearn"

# Subject ID from array task ID (formatted as 01, 02, etc.)
export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

# Run the deface script
python deface.py ${PARTICIPANT_LABEL} --bids_folder ${BIDS_FOLDER}
