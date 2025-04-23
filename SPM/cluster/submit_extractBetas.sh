#!/bin/bash
#SBATCH --job-name=extractBetas
#SBATCH --output=/home/ecasim/logs/extractBetas_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem-per-cpu=32G
#SBATCH --time=15:00:00

module load anaconda3
module load matlab
source /home/${USER}/.bashrc
source activate nipype

python $HOME/git/multlearn-sns/SPM/code/extractBetas.py