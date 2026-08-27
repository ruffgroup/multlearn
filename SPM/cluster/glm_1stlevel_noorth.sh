#!/bin/bash
#SBATCH --job-name=GLM_1st_noorth
#SBATCH --output=/home/gdehol/logs/GLM_1st_noorth_%A-%a.txt
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --time=20:00:00

# First level WITHOUT SPM's within-condition serial orthogonalisation.
# Writes to nipype/<model>_noorth so the original results are untouched.
#   sbatch --array=1-64 SPM/cluster/glm_1stlevel_noorth.sh model7
module load matlab/r2023b          # also puts apptainer on PATH for the wrapper
PY=$HOME/data/conda/envs/multlearn/bin/python
MODEL=${1:-model7}
PARTICIPANT_LABEL=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})

$PY $HOME/git/multlearn-sns/SPM/code/GLM_1stlevel_nipype.py \
    /shares/zne.uzh/multlearn \
    --base_dir /shares/zne.uzh/multlearn \
    --model $MODEL \
    --no-orth \
    --mask /shares/zne.uzh/multlearn/mask_ICV.nii \
    --bestfitting_path /shares/zne.uzh/multlearn/fittedParametersRecoveredModels/bestFittingVals \
    --mlab_path $(command -v matlab) \
    --spm_path ~/spm12 \
    --Nslices 40 \
    --refSlice 20 \
    --participant_label ${PARTICIPANT_LABEL}
