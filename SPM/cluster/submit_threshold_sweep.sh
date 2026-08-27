#!/bin/bash
#SBATCH --job-name=threshold_sweep
#SBATCH --output=/home/gdehol/logs/threshold_sweep_%A-%a.txt
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --time=10:00:00

# Usage (submit from a LOGIN shell -- `module load matlab` silently fails otherwise):
#   sbatch --array=0-4 SPM/cluster/submit_threshold_sweep.sh snpm
#   sbatch --array=0-4 SPM/cluster/submit_threshold_sweep.sh nilearn
MODE=${1:-nilearn}
PY=$HOME/data/conda/envs/multlearn/bin/python
CODE=$HOME/git/multlearn-sns/SPM/code

if [ "$MODE" = "snpm" ]; then
    # pin the MATLAB the original 2nd level was run with; the module also
    # pulls in apptainer, which the matlab wrapper needs on PATH
    module load matlab/r2023b
    export MATLAB_CMD=$(command -v matlab)
    $PY $CODE/snpm_threshold_sweep.py --index ${SLURM_ARRAY_TASK_ID:-0}
else
    # connectivity 18 matches SPM's spm_clusters and forces n_jobs=1
    $PY $CODE/nilearn_threshold_sweep.py \
        --index ${SLURM_ARRAY_TASK_ID:-0} \
        --n-perm 5000 \
        --connectivity 18
fi
