import subprocess
import os
import numpy as np

output_prefix = '/mnt/d/multlearn-sns/Modelling/Decoding/run_surf2img/out_'
bids_folder = "/mnt/d/data/ds-mlearn"
derivatives = os.path.join(bids_folder, "derivatives")
IDs = range(1,65)
for ID in IDs:
    if ID not in [8, 13, 16, 31, 32, 44]:
        subject_path = f'/mnt/d/data/ds-mlearn/derivatives/freesurfer/sub-{ID:02d}'
        path_in = f'/mnt/d/data/ds-mlearn/derivatives/freesurfer/sub-{ID:02d}/surf'
        path_out = f'/mnt/d/data/ds-mlearn/derivatives/encoding_model/sub-{ID:02d}/anat'
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        output_filename = f'sub-{ID:02d}_surf2img_v1.nii.gz'
        output_file = os.path.join(path_out, output_filename)
        lh_file = os.path.join(path_in, 'lh.benson14_varea.mgz')
        rh_file = os.path.join(path_in, 'rh.benson14_varea.mgz')
        affine_file = os.path.join(derivatives,"fmriprep",f"sub-{ID:02d}", "func", f"sub-{ID:02d}_task-learn_run-1_space-T1w_desc-brain_mask.nii.gz")
        process_output_file = output_prefix + str(ID) + '.txt'
        with open(process_output_file, 'w') as f:
            process = subprocess.run(['python', '-m', 'neuropythy', 'surface_to_image', str(subject_path), str(output_file), '--lh=' + str(lh_file), '--rh=' + str(rh_file), '--method=nearest', '--image=' + str(affine_file)], stdout=f, stderr=subprocess.STDOUT)
        print('Finished subject', ID)