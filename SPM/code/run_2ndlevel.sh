export DATA_FOLDER=/shares/zne.uzh/

python $HOME/git/multlearn-sns/SPM/code/GLM_2ndlevel_nipype.py \
    17 \
    /shares/zne.uzh/multlearn/nipype/model2 \
    --model fmri \
    --inference cluster \
    --base_dir /shares/zne.uzh/multlearn/nipype/model2 \
    --mlab_path /apps/opt/containers/bin/matlab/r2023b/matlab \
    --spm_path ~/spm12 \
    --tVal 3.1
    # --tVal 2.8 3.1 4 5 7 8 9 10 12