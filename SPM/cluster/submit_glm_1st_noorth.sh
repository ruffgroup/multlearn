#!/bin/bash
# Submit the non-orthogonalised first level for the GLM subject set.
# Run from inside SPM/cluster/ (glm_1stlevel_noorth.sh is referenced relatively).
participants=($(seq -s ' ' 1 64 | tr ' ' '\n' | grep -v -E '^(8|13|16|31|32|44)$' | tr '\n' ',' | sed 's/,$//'))
sbatch --array=${participants} glm_1stlevel_noorth.sh ${1:-model7}
