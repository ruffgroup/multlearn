import argparse
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, traits, File, Directory, TraitedSpec
from nipype.interfaces.matlab import MatlabCommand
import nipype.interfaces.matlab as matlab
from pathlib import Path
from nilearn import image
import nibabel as nib
import numpy as np
import nipype.pipeline.engine as pe
import os
import os.path as op


class SpmVOIInputSpec(BaseInterfaceInputSpec):
    spm_path = Directory(exists=True, desc='SPM directory', mandatory=True)
    spm_mat_file = File(exists=True, desc='Path to SPM.mat file', mandatory=True)
    output_dir = Directory(exists=True, desc='Output directory', mandatory=True)
    contrast = traits.Int(desc='Contrast number', mandatory=True)
    run = traits.Int(desc='Run number', mandatory=True)
    peak_coords = traits.List(traits.Float, desc='Peak coordinates [x, y, z]', mandatory=True)
    radius = traits.Float(desc='Radius for the spherical VOI', mandatory=True)
    voi_name = traits.Str(desc='name for VOI', mandatory=True)

class SpmVOIOutputSpec(TraitedSpec):
    voi_mat_file = File(exists=True, desc='Path to the output VOI.mat file')

class SpmVOI(BaseInterface):
    input_spec = SpmVOIInputSpec
    output_spec = SpmVOIOutputSpec
    _jobtype = "util"
    _jobname = "voi"

    def _run_interface(self, runtime):
        spm_path = self.inputs.spm_path
        spm_mat_file = self.inputs.spm_mat_file
        peak_coords = self.inputs.peak_coords
        voi_name = self.inputs.voi_name
        contrast = self.inputs.contrast
        run = self.inputs.run
        radius = self.inputs.radius

        matlab_script = f"""
        addpath('{spm_path}');
        spm('defaults', 'FMRI');
        spm_jobman('initcfg');
        matlabbatch{{1}}.spm.util.voi.spmmat = {{'{spm_mat_file}'}};
        matlabbatch{{1}}.spm.util.voi.adjust = NaN;
        matlabbatch{{1}}.spm.util.voi.session = {run};
        matlabbatch{{1}}.spm.util.voi.name = '{voi_name}';
        matlabbatch{{1}}.spm.util.voi.roi{{1}}.spm.spmmat = {{''}};
        matlabbatch{{1}}.spm.util.voi.roi{{1}}.spm.contrast = {contrast};
        matlabbatch{{1}}.spm.util.voi.roi{{1}}.spm.conjunction = 1;
        matlabbatch{{1}}.spm.util.voi.roi{{1}}.spm.threshdesc = 'none';
        matlabbatch{{1}}.spm.util.voi.roi{{1}}.spm.thresh = 0.05;
        matlabbatch{{1}}.spm.util.voi.roi{{1}}.spm.extent = 0;
        matlabbatch{{1}}.spm.util.voi.roi{{1}}.spm.mask = struct('contrast', {{}}, 'thresh', {{}}, 'mtype', {{}});
        matlabbatch{{1}}.spm.util.voi.roi{{2}}.sphere.centre = {peak_coords};
        matlabbatch{{1}}.spm.util.voi.roi{{2}}.sphere.radius = 20;
        matlabbatch{{1}}.spm.util.voi.roi{{2}}.sphere.move.fixed = 1;
        matlabbatch{{1}}.spm.util.voi.roi{{3}}.sphere.centre = [0, 0, 0];
        matlabbatch{{1}}.spm.util.voi.roi{{3}}.sphere.radius = {radius};
        matlabbatch{{1}}.spm.util.voi.roi{{3}}.sphere.move.global.spm = 1;
        matlabbatch{{1}}.spm.util.voi.roi{{3}}.sphere.move.global.mask = 'i2';
        matlabbatch{{1}}.spm.util.voi.expression = 'i1 & i3';
        spm_jobman('run', matlabbatch);
        clear matlabbatch
        """

        script_file = os.path.join(runtime.cwd, 'extract_voi.m')
        with open(script_file, 'w') as f:
            f.write(matlab_script)

        mlab = matlab.MatlabCommand(script=matlab_script, paths=[spm_path])
        result = mlab.run()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['voi_mat_file'] = os.path.join(''.join(self.inputs.spm_mat_file.split("SPM.mat")[:-1]), 'VOI_'+self.inputs.voi_name+"_"+str(self.inputs.run)+'.mat')
        return outputs


def get_peak_mni_coordinates(img):
    """
    Find the peak voxel in a NIfTI image and return its MNI coordinates.

    Parameters:
    nifti_file (str): Path to the NIfTI file.

    Returns:
    tuple: MNI coordinates of the peak voxel (x, y, z).
    """
    # Load the NIfTI image
    data = img.get_fdata()

    # Find the coordinates of the peak voxel
    peak_coords = np.unravel_index(np.nanargmax(data), data.shape)

    # Convert voxel coordinates to MNI coordinates
    peak_mni = nib.affines.apply_affine(img.affine, peak_coords)

    return peak_mni

def main(subject, roi, model, contrast, data_folder, mlab_path, spm_path, direction, work_dir):
    MatlabCommand.set_default_paths(spm_path)
    MatlabCommand.set_default_matlab_cmd(mlab_path)
    data_folder = Path(data_folder)

    tmap = data_folder / 'nipype' / model / '2ndLevel' / f'cluster_SnPM_SecondLevel_con{contrast}' / f'SnPM_filtered_t2_8_{direction}.nii'
    roi_mask = data_folder / 'nipype' / model / 'ROI' / f'{roi}.nii'
    spm_mat_file = data_folder / 'nipype' / model / '1stLevel' / f'sub-{subject:02d}' / 'SPM.mat'

    # Find the peak coordinate within the masked ROI
    print(tmap)
    masked_tmap = image.math_img("img1 * img2", img1=tmap, img2=roi_mask)
    peak_coords = get_peak_mni_coordinates(masked_tmap)
    print(f"Peak coordinates: {peak_coords}")


    target_dir = data_folder / 'nipype' / model / 'timeseries' / f'sub-{subject:02d}'
    target_dir.mkdir(parents=True, exist_ok=True)

    voi_nodes = []

    for run_id in range(1, 7):
        voi_node = pe.Node(SpmVOI(spm_path=spm_path,
                    spm_mat_file=spm_mat_file,
                    peak_coords=[peak_coords[0].astype(float), peak_coords[1].astype(float), peak_coords[2].astype(float)],
                    output_dir=target_dir,
                    contrast=int(contrast),
                    run=run_id,
                    radius=6,
                    voi_name=f'{roi}_{contrast}_{run_id}'),
                    name=f'voi_node_subject_run_{run_id}')
        
    # Create a workflow
    workflow = pe.Workflow(name=f'voi_workflow_subject-{subject}_roi-{roi}_contrast-{contrast}', base_dir=work_dir)
    workflow.add_nodes([voi_node])
    workflow.run() 




if __name__ == '__main__':

    parser = argparser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, type=int)
    parser.add_argument('--ROI', default='A1_L')
    parser.add_argument('--model', default='model2')
    parser.add_argument('--contrast', default=17)
    # parser.add_argument("--data_folder", type=str, default="/shares/zne.uzh/multlearn")
    parser.add_argument("--data_folder", type=str, default="/data/ds-mlearn")
    parser.add_argument("--mlab_path", type=str, default="/apps/opt/containers/bin/matlab/r2023b/matlab")
    parser.add_argument("--spm_path", type=str, default=op.join(os.environ['HOME'], 'spm12'))
    parser.add_argument("--direction", default="pos")
    parser.add_argument("--work_dir", default="/tmp/working_dir")

    args = parser.parse_args()


    main(subject=args.subject, roi=args.ROI, model=args.model, contrast=args.contrast, data_folder=args.data_folder, mlab_path=args.mlab_path, spm_path=args.spm_path,
         direction=args.direction, work_dir=args.work_dir)