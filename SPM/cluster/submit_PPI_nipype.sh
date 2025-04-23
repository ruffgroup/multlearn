#!/bin/bash
#SBATCH --job-name=PPI_nipype
#SBATCH --output=/home/ecasim/logs/PPI_nipype_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem-per-cpu=32G
#SBATCH --time=15:00:00

module load anaconda3
module load matlab
source /home/${USER}/.bashrc
source activate nipype

python $HOME/git/multlearn-sns/SPM/code/PPI_nipype.py /shares/zne.uzh/multlearn/nipype/model2/ROI/cluster_4_con1_4_0_neg.nii --model model2 --data_folder /shares/zne.uzh/multlearn --mlab_path /apps/opt/containers/bin/matlab/r2023b/matlab --spm_path ~/data/spm12