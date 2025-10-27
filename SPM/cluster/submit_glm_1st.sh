#!/bin/bash


# Define the participants (excluding 8, 13, 16, 31, 32, 44)
participants=($(seq -s ' ' 1 64 | tr ' ' '\n' | grep -v -E '^(8|13|16|31|32|44)$' | tr '\n' ',' | sed 's/,$//'))

# Submit the job script with the ROI and variable as arguments
sbatch --array=${participants} glm_1stlevel.sh model7