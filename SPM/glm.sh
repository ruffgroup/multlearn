#!/bin/bash
#SBATCH --job-name=glm
#SBATCH --output=/home/%u/logs/res_glm_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
module load matlab
export SINGULARITYENV_TEMPLATEFLOW_HOME=/opt/templateflow
matlab -batch "run code/MASTER_fMRI_analysis.m"