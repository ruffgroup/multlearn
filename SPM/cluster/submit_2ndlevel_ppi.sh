#!/bin/bash

# Define the ROIs and variables
# ROIS=("S1_R")
# ROIs=("A1_L" "A1_R")
# ROIs=("S1_R" "A1_R")
# variables=("urpe" "surprise" "choice" "feedback")

ROIs=("AG_RPE_L" "AG_S_L" "AG_S_R" "V1_RPE_L" "DLPFC_S_L")
variables=("urpe" "surprise")

# Loop through each ROI and variable combination
for ROI in "${ROIs[@]}"; do
    for variable in "${variables[@]}"; do
        # Submit the job script with the ROI and variable as arguments
        sbatch --array=1-7 2nd_level_ppi.sh "$ROI" "$variable"
    done
done
