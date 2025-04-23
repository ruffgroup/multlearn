#!/bin/bash
#SBATCH --job-name=GLM_1stlevel_nipype
#SBATCH --output=/home/ecasim/logs/GLM_1stlevel_nipype_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem-per-cpu=32G
#SBATCH --time=20:00:00

module load anaconda3
module load matlab
source /home/${USER}/.bashrc
source activate nipype

python $HOME/git/multlearn-sns/SPM/code/GLM_1stlevel_nipype.py /shares/zne.uzh/multlearn --base_dir /shares/zne.uzh/multlearn --model PPI --mask /shares/zne.uzh/multlearn/mask_ICV.nii --ppi_mask /shares/zne.uzh/multlearn/nipype/model2/ROI/cluster_4_con15_8_0_neg.nii --bestfitting_path /shares/zne.uzh/multlearn/fittedParametersRecoveredModels/bestFittingVals --mlab_path /apps/opt/containers/bin/matlab/r2023b/matlab --spm_path ~/data/spm12 --Nslices 40 --refSlice 20