export DATA_FOLDER=/shares/zne.uzh/

python GLM_1stlevel_nipype.py \
    $DATA_FOLDER/multlearn \
    --base_dir $DATA_FOLDER \
    --model model2 \
    --mask $DATA_FOLDER/multlearn/mask_ICV.nii \
    --bestfitting_path $DATA_FOLDER/multlearn/fittedParametersRecoveredModels/bestFittingVals \
    --mlab_path /apps/opt/containers/bin/matlab/r2023b/matlab \
    --spm_path ~/spm12 \
    --Nslices 40 \
    --refSlice 20 \
    --participant_label 03  
