#!/bin/bash

# Define the ROIs and variables
# ROIs=("A1_L" "A1_R" "S1_R")
# ROIs=("A1_L" "A1_R")
ROIs=("S1_R" "A1_R")
ROIs=("AG_RPE_L" "AG_S_L" "AG_S_R" "V1_RPE_L" "DLPFC_S_L")
variables=("urpe" "rpe" "surprise" "choice" "feedback")
variables=("urpe" "surprise")

# Define the participants (excluding 8, 13, 16, 31, 32, 44)
participants=($(seq -s ' ' 1 64 | tr ' ' '\n' | grep -v -E '^(8|13|16|31|32|44)$' | tr '\n' ',' | sed 's/,$//'))

# Loop through each ROI and variable combination
for ROI in "${ROIs[@]}"; do
    for variable in "${variables[@]}"; do
        # Submit the job script with the ROI and variable as arguments
        sbatch --array=${participants} glm_1stlevel_ppi.sh "$ROI" "$variable"
    done
done
