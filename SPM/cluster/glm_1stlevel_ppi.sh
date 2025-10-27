#!/bin/bash
#SBATCH --job-name=GLM_1stlevel_nipype_ppi
#SBATCH --output=/home/gdehol/logs/GLM_1stlevel_nipype_ppi_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --time=30:00

. $HOME/init_conda.sh

module load matlab
module unload python
# source /home/${USER}/.bashrc
# source activate nipype
conda activate multlearn

# Zero-pad the SLURM_ARRAY_TASK_ID to two digits
PARTICIPANT_LABEL=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})

ROI=$1
VARIABLE=$2

python $HOME/git/multlearn-sns/SPM/code/GLM_1stlevel_nipype.py \
    /shares/zne.uzh/multlearn \
    --base_dir /shares/zne.uzh/multlearn \
    --model PPI \
    --mask /shares/zne.uzh/multlearn/mask_ICV.nii \
    --bestfitting_path /shares/zne.uzh/multlearn/fittedParametersRecoveredModels/bestFittingVals \
    --mlab_path /apps/opt/containers/bin/matlab/r2023b/matlab \
    --spm_path ~/spm12 \
    --Nslices 40 \
    --refSlice 20 \
    --participant_label ${PARTICIPANT_LABEL} \
    --ppi_roi ${ROI} \
    --ppi_variable ${VARIABLE}