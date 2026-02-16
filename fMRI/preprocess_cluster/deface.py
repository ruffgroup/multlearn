from pathlib import Path
import pydeface.utils as pdu
import argparse

def main(subject, bids_folder='/data/ds-multlearn'):

    subject_folder = Path(bids_folder) / f'sub-{subject:02d}'
    anat_folder = subject_folder / 'anat'
    func_folder = subject_folder / 'func'


    t1w_images = list(anat_folder.glob('*_T1w.nii'))

    assert(len(list(t1w_images)) == 1), f"Expected exactly one T1w image in {anat_folder}, found {len(list(t1w_images))}"

    in_file = list(t1w_images)[0]
    out_file = anat_folder / f'sub-{subject:02d}_T1w.nii'

    pdu.deface_image(in_file, out_file, forcecleanup=True, force=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deface MRI images for a given subject.')
    parser.add_argument('subject', type=int, help='Subject ID (e.g., 01)')
    parser.add_argument('--bids_folder', type=str, default='/data/ds-multlearn', help='Path to the BIDS dataset folder')
    
    args = parser.parse_args()
    main(args.subject, args.bids_folder)