#!/bin/bash
#SBATCH --job-name=fmriprep
#SBATCH --output=/home/%u/logs/res_fmriprep_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
module load singularityce/3.10.2
export SINGULARITYENV_FS_LICENSE=$HOME/freesurfer/license.txt
export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
export SINGULARITYENV_TEMPLATEFLOW_HOME=/opt/templateflow
singularity run -u -B /shares/zne.uzh/ds-mlearn:/data -B /scratch/$USER/workflow_folders:/workflow -B /scratch/$USER/templateflow:/opt/templateflow --cleanenv /shares/hare.econ.uzh/containers/fmriprep_20.2.3  /data /data/derivatives participant --participant-label $PARTICIPANT_LABEL  --output-spaces MNI152NLin2009cAsym T1w fsaverage fsnative  --dummy-scans 5 --skip_bids_validation -w /workflow --no-submm-recon --fmap-bspline
