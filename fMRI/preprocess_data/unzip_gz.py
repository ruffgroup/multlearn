import os
import os.path as op
import shutil
import gzip
import glob
import argparse

def main(subject, fmriprep_folder='C:/Users/ellac/ds-mlearn/derivatives/fmriprep'):
    if int(subject) < 10:
        zipped_funcs = glob.glob(op.join(fmriprep_folder, f"sub-0{subject}", "func", '*nii.gz'))
        zipped_anat = glob.glob(op.join(fmriprep_folder, f"sub-0{subject}", "anat", '*nii.gz'))
    else:
        zipped_funcs = glob.glob(op.join(fmriprep_folder, f"sub-{subject}", "func", '*nii.gz'))
        zipped_anat = glob.glob(op.join(fmriprep_folder, f"sub-{subject}", "anat", '*nii.gz'))

    for func in zipped_funcs:
        with gzip.open(func, 'r') as f_in, open(func.split('.')[0]+".nii", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    for anat in zipped_anat:
         with gzip.open(anat, 'r') as f_in, open(anat.split('.')[0]+".nii", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=int)
    parser.add_argument('--fmriprep_folder', default='C:/Users/ellac/ds-mlearn/derivatives/fmriprep')
    args = parser.parse_args()

    main(args.subject, fmriprep_folder=args.fmriprep_folder)