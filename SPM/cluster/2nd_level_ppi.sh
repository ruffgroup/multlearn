#!/bin/bash
#SBATCH --job-name=GLM_2ndlevel_nipype
#SBATCH --output=/home/gdehol/logs/GLM_2ndlevel_nipype_ppi_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=1:00:00

module load anaconda3
module load matlab
source /home/${USER}/.bashrc
conda activate multlearn

ROI=$1
VARIABLE=$2
SVC_ROI=$3

# Start building the Python command
PYTHON_CMD="python $HOME/git/multlearn-sns/SPM/code/GLM_2ndlevel_nipype.py \
    $SLURM_ARRAY_TASK_ID \
    /shares/zne.uzh/multlearn \
    --model fmri \
    --inference cluster \
    --base_dir \"/shares/zne.uzh/multlearn/nipype/PPI/${ROI}_${VARIABLE}\" \
    --mlab_path /apps/opt/containers/bin/matlab/r2023b/matlab \
    --spm_path ~/spm12 \
    --tVal 2.8 3.1 4"

# Append the --roi argument if SVC_ROI is provided
if [ -n "$SVC_ROI" ]; then
    PYTHON_CMD="$PYTHON_CMD --roi $SVC_ROI"
fi

# Execute the Python command
eval $PYTHON_CMD