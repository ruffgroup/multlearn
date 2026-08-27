#!/bin/bash
# Pull the (small) threshold-sweep result maps back for local plotting.
# Aggregation runs on the cluster; only summary-sized maps come south.
set -e
DEST=/Users/gdehol/git/multlearn-sns/notes/data/threshold_sweep
mkdir -p "$DEST"
rsync -av --prune-empty-dirs \
    --include='*/' \
    --include='*.nii' --include='*.nii.gz' --include='*.tsv' --include='meta.json' \
    --exclude='*' \
    sciencecluster:/shares/zne.uzh/multlearn/threshold_sweep/ "$DEST/"
du -sh "$DEST"
