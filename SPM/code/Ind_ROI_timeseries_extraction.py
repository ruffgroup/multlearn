import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
import os
from glob import glob
from nilearn.maskers import NiftiSpheresMasker
from nilearn.image import get_data, math_img
from scipy.ndimage import center_of_mass
from bids.layout import BIDSLayout
import argparse
import re

import os
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, traits, File, Directory, TraitedSpec
import nipype.interfaces.matlab as matlab
from nipype.interfaces.matlab import MatlabCommand
from nipype import Node, Workflow
import shutil

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


# Define a function to find the peak coordinate within the masked ROI
def find_peak_coordinate_within_mask(roi_img, t_img):
    roi_data = get_data(roi_img)
    t_data = get_data(t_img)
    
    # Mask the t-contrast image with the ROI
    masked_t_data = np.where(roi_data != 0, t_data, 0)
    
    # Find the index of the peak value within the masked t-contrast image
    peak_index = np.argmax(np.abs(masked_t_data))
    
    # Convert this index back to the coordinate within the image
    peak_coord = np.unravel_index(peak_index, t_data.shape)
    return peak_coord

# Define a function to find the peak coordinate within the larger ROI for an individual subject
def find_peak_coordinate(roi_img, fmri_img):
    roi_data = get_data(roi_img)
    fmri_data = get_data(fmri_img)

    # Get the masked fMRI data within the ROI
    masked_data = fmri_data[roi_data > 0]

    # Find the index of the peak value within the masked data
    peak_index = np.argmax(np.abs(masked_data))
    
    # Convert this index back to the coordinate within the ROI
    peak_coord = np.unravel_index(peak_index, roi_data.shape)
    return peak_coord


def main(roi_mask, model="model2", data_folder="/shares/zne.uzh/multlearn", mlab_path="/usr/local/MATLAB/R2022b/bin/matlab", spm_path="~/spm12", source="fullbrain"):
    MatlabCommand.set_default_paths(spm_path)
    MatlabCommand.set_default_matlab_cmd(mlab_path)
    # Load the group-level ROI mask
    group_roi_img = nib.load(roi_mask)

    # Load the group-level ROI mask and group-level t-contrast image

    # cluster_1_con17_3_1_neg.nii.gz


    con = roi_mask.split("/")[-1].split("_")[-4]
    temp = re.compile("([a-zA-Z]+)([0-9]+)")
    con_nr = temp.match(con).groups()[1]
    t_val = roi_mask.split("/")[-1].split(str(con)+'_')[-1]
    print(t_val)
    group_t_img = nib.load(op.join(data_folder, "nipype", model, "2ndLevel/cluster_SnPM_SecondLevel_con"+str(con_nr)+"/SnPM_filtered_t"+str(t_val)))
    # Find the peak coordinate within the group-level ROI
    # Find the peak coordinate within the group-level ROI masked t-contrast image
    peak_coord = find_peak_coordinate_within_mask(group_roi_img, group_t_img)
    
    output_dir = ''.join(roi_mask.split("/")[-1].split("_")[2])+"/"+''.join(roi_mask.split("/")[-1].split("_")[:2])+'_'+roi_mask.split("/")[-1].split("_")[-1].split(".")[0]

    subject_list = range(1,65)
    subject_ids = [str(sub).zfill(2) for sub in subject_list if sub not in [8, 13, 16, 31, 32, 44]]

    #subject_ids = ["46"]
    # Iterate over individual subject fMRI images
    for subject_id in subject_ids:
        check_betas = op.join(data_folder, "nipype", model, "1stLevel/sub-"+subject_id,"beta*.nii")
        fn_check = glob(check_betas)
        if len(fn_check) == 0:
            betas_path = (
            op.join(data_folder,'nipype',f"workingdir_{model}","first_level_wf",f"_subject_id_{subject_id}/","level1estimate","beta*.nii")
            )
            fn = glob(betas_path)
            for f in fn:
                shutil.copy(f, op.join(data_folder, "nipype", model, "1stLevel/sub-"+subject_id+"/"))

        output_path = op.join(data_folder, 'nipype', 'PPI',model, output_dir, f'sub-{subject_id}')
        if not op.exists(output_path):
                os.makedirs(output_path)

        for run_id in range(1, 7):
  
            voi_node = Node(SpmVOI(spm_path=spm_path,
                       spm_mat_file=op.join(data_folder, "nipype", model, "1stLevel/sub-"+subject_id,"SPM.mat"),
                       peak_coords=[peak_coord[0].astype(float), peak_coord[1].astype(float), peak_coord[2].astype(float)],
                       output_dir=output_path,
                       contrast=int(con_nr),
                       run=run_id,
                       radius=6,
                       voi_name=roi_mask.split("/")[-1].split(".")[-2]), name='voi_node')
            

            # Create a workflow
            workflow = Workflow(name='voi_workflow', base_dir=output_path)
            workflow.add_nodes([voi_node])
            try:
            # Run the workflow
                workflow.run() 
            except:
                continue
            

        print(f'Ran VOI extraction for {subject_id}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("roi_mask", type=str)
    parser.add_argument("--model", type=str, default="model2")
    parser.add_argument("--data_folder", type=str, default="/shares/zne.uzh/multlearn")
    parser.add_argument("--mlab_path", type=str, default="/apps/opt/containers/bin/matlab/r2023b/matlab")
    parser.add_argument("--spm_path", type=str, default="~/spm12")

    args = parser.parse_args()

    main(args.roi_mask, model = args.model, data_folder = args.data_folder, mlab_path = args.mlab_path, spm_path = args.spm_path)

