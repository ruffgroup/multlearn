#!/bin/bash
#SBATCH --job-name=GLM_2ndlevel_nipype
#SBATCH --output=/home/gdehol/logs/GLM_2ndlevel_nipype_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --time=5:00:00

module load anaconda3
module load matlab
source /home/${USER}/.bashrc
# source activate nipype
conda activate multlearn

python $HOME/git/multlearn-sns/SPM/code/GLM_2ndlevel_nipype.py \
    $SLURM_ARRAY_TASK_ID \
    /shares/zne.uzh/multlearn/ \
    --model fmri \
    --inference cluster \
    --base_dir /shares/zne.uzh/multlearn/nipype/model7 \
    --mlab_path /apps/opt/containers/bin/matlab/r2023b/matlab \
    --spm_path ~/spm12 \
    --tVal 2.8 3.1 4 

# python $HOME/git/multlearn-sns/SPM/code/GLM_2ndlevel_nipype.py$SLURM_ARRAY_TASK_ID /shares/zne.uzh/multlearn --model fmri --inference cluster --base_dir /shares/zne.uzh/multlearn/nipype/model2/PPI/con1/cluster4_neg --mlab_path /apps/opt/containers/bin/matlab/r2023b/matlab --spm_path ~/data/spm12 --tVal 2.8 3.1 4 5 7 8 9 10 12