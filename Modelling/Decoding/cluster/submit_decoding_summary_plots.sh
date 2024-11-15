#!/bin/bash
#SBATCH --job-name=decoding_summary_plots
#SBATCH --output=/home/ecasim/logs/decoding_summary_plots_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --time=30:00

module load anaconda3
source /home/${USER}/.bashrc
source activate multlearn-decoding

python $HOME/git/multlearn-sns/Modelling/Decoding/decoding_summary_plots.py --bids_folder /shares/zne.uzh/multlearn/ds-mlearn --paramregr_folder /shares/zne.uzh/multlearn --space 'T1w' --mask 'v1' --nvoxels 400