#!/bin/bash
#SBATCH --job-name=GLM_1stlevel_nipype
#SBATCH --output=/home/gdehol/logs/GLM_1stlevel_nipype_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --time=20:00:00

. $HOME/init_conda.sh

module load matlab
module unload python
# source /home/${USER}/.bashrc
# source activate nipype
conda activate multlearn

# Zero-pad the SLURM_ARRAY_TASK_ID to two digits
PARTICIPANT_LABEL=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})

#If not defined, use model6
if [ -z "$1" ]; then
    echo "No model specified, using default model6"
    MODEL="model6"
else
    echo "Using specified model: $1"
    MODEL=$1
fi

python $HOME/git/multlearn-sns/SPM/code/GLM_1stlevel_nipype.py \
    /shares/zne.uzh/multlearn \
    --base_dir /shares/zne.uzh/multlearn \
    --model $MODEL \
    --mask /shares/zne.uzh/multlearn/mask_ICV.nii \
    --bestfitting_path /shares/zne.uzh/multlearn/fittedParametersRecoveredModels/bestFittingVals \
    --mlab_path /apps/opt/containers/bin/matlab/r2023b/matlab \
    --spm_path ~/spm12 \
    --Nslices 40 \
    --refSlice 20 \
    --participant_label ${PARTICIPANT_LABEL}