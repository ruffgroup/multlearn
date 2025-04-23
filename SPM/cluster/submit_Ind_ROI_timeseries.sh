#!/bin/bash
#SBATCH --job-name=GLM_1stlevel_nipype
#SBATCH --output=/home/ecasim/logs/Ind_ROI_timeseries_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem-per-cpu=32G
#SBATCH --time=15:00:00

module load anaconda3
module load matlab
source /home/${USER}/.bashrc
source activate nipype

python $HOME/git/multlearn-sns/SPM/code/Ind_ROI_timeseries_extraction.py /shares/zne.uzh/multlearn/nipype/model2/ROI/cluster_9_con15_8_0_neg.nii --model model2 --data_folder /shares/zne.uzh/multlearn --mlab_path /apps/opt/containers/bin/matlab/r2023b/matlab --spm_path ~/data/spm12