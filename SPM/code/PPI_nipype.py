import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
import os
from glob import glob
from bids.layout import BIDSLayout
import argparse
import re

import os
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, traits, File, Directory, TraitedSpec
import nipype.interfaces.matlab as matlab
from nipype.interfaces.matlab import MatlabCommand
from nipype import Node, Workflow

class SpmPPIInputSpec(BaseInterfaceInputSpec):
    spm_path = Directory(exists=True, desc='SPM directory', mandatory=True)
    spm_mat_file = File(exists=True, desc='Path to SPM.mat file', mandatory=True)
    voi_file = File(exists=True, desc='Path to VOI.mat file', mandatory=True)
    learning_type = traits.Str(desc='rpe or surprise', mandatory=True)
    ppi_name = traits.Str(desc='name for PPI', mandatory=True)
    run = traits.Int(desc='Run number', mandatory=True)
    model = traits.Str(desc='model', mandatory=True)

class SpmPPIOutputSpec(TraitedSpec):
    ppi_file = File(exists=True, desc='Path to the output VOI.mat file')

class SpmPPI(BaseInterface):
    input_spec = SpmPPIInputSpec
    output_spec = SpmPPIOutputSpec
    _jobtype = "util"
    _jobname = "voi"

    def _run_interface(self, runtime):
        spm_path = self.inputs.spm_path
        spm_mat_file = self.inputs.spm_mat_file
        voi_file = self.inputs.voi_file
        learning_type = self.inputs.learning_type
        ppi_name = self.inputs.ppi_name
        run = self.inputs.run
        model = self.inputs.model
        
        if model == "model2":
            matlab_script = f"""
            addpath('{spm_path}');
            spm('defaults', 'FMRI');
            spm_jobman('initcfg');
            matlabbatch{{1}}.spm.stats.ppi.spmmat = {{'{spm_mat_file}'}};
            matlabbatch{{1}}.spm.stats.ppi.type.ppi.voi = {{'{voi_file}'}};
            if contains('{learning_type}', 'surprise') 
                matlabbatch{{1}}.spm.stats.ppi.type.ppi.u = [1 1 0
                    1 2 1
                    1 3 0
                    2 1 0
                    2 2 0];
            elseif contains('{learning_type}', 'rpe')
                matlabbatch{{1}}.spm.stats.ppi.type.ppi.u = [1 1 0
                    1 2 0
                    1 3 0
                    2 1 0
                    2 2 1];
            end
            matlabbatch{{1}}.spm.stats.ppi.name = '{ppi_name}';
            matlabbatch{{1}}.spm.stats.ppi.disp = 0;
            spm_jobman('run', matlabbatch);
            clear matlabbatch
            """
        elif model == "model1":
            matlab_script = f"""
            addpath('{spm_path}');
            spm('defaults', 'FMRI');
            spm_jobman('initcfg');
            matlabbatch{{1}}.spm.stats.ppi.spmmat = {{'{spm_mat_file}'}};
            matlabbatch{{1}}.spm.stats.ppi.type.ppi.voi = {{'{voi_file}'}};
            if contains('{learning_type}', 'surprise') 
                matlabbatch{{1}}.spm.stats.ppi.type.ppi.u = [1 1 0
                    1 2 1
                    2 1 0
                    2 2 0];
            elseif contains('{learning_type}', 'rpe')
                matlabbatch{{1}}.spm.stats.ppi.type.ppi.u = [1 1 0
                    1 2 0
                    2 1 0
                    2 2 1];
            end
            matlabbatch{{1}}.spm.stats.ppi.name = '{ppi_name}';
            matlabbatch{{1}}.spm.stats.ppi.disp = 0;
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
        outputs['ppi_file'] = os.path.join(''.join(self.inputs.spm_mat_file.split("SPM.mat")[:-1]), 'PPI_'+self.inputs.ppi_name+'.mat')
        return outputs
    
def main(roi_mask, model="model2", data_folder="/shares/zne.uzh/multlearn", mlab_path="/usr/local/MATLAB/R2022b/bin/matlab", spm_path="~/spm12"):
    MatlabCommand.set_default_paths(spm_path)
    MatlabCommand.set_default_matlab_cmd(mlab_path)

    con = roi_mask.split("/")[-1].split("_")[-4]
    temp = re.compile("([a-zA-Z]+)([0-9]+)")
    con_nr = temp.match(con).groups()[1]

    output_dir = ''.join(roi_mask.split("/")[-1].split("_")[2])+"/"+''.join(roi_mask.split("/")[-1].split("_")[:2])+'_'+roi_mask.split("/")[-1].split("_")[-1].split(".")[0]
    layout = BIDSLayout(op.join(data_folder,"ds-mlearn/"), derivatives=True)
    # list of subject identifiers
    subject_ids = layout.get_subjects()
    subject_ids = [sub for sub in subject_ids if int(sub) not in [8, 13, 16, 31, 32, 44]]
    for subject_id in subject_ids:
        output_path = op.join(data_folder, 'nipype', model, 'PPI', output_dir, f'sub-{subject_id}')
        for run_id in range(1,7):
            if op.exists(op.join(data_folder, "nipype", model, "1stLevel/sub-"+subject_id,"VOI_"+roi_mask.split("/")[-1].split(".")[-2]+"_"+str(run_id)+'.mat')):

                if int(con_nr) in [1, 22]:
                    PPI_node = Node(SpmPPI(spm_path=spm_path,
                                spm_mat_file=op.join(data_folder, "nipype",model, "1stLevel/sub-"+subject_id,"SPM.mat"),
                                voi_file=op.join(data_folder, "nipype",model,"1stLevel/sub-"+subject_id,"VOI_"+roi_mask.split("/")[-1].split(".")[-2]+"_"+str(run_id)+'.mat'),
                                learning_type="rpe",
                                run=run_id,
                                ppi_name=roi_mask.split("/")[-1].split(".")[-2]+"_"+str(run_id),
                                 model=model), name='ppi_node')
                    
                else:
                    PPI_node = Node(SpmPPI(spm_path=spm_path,
                                spm_mat_file=op.join(data_folder, "nipype",model, "1stLevel/sub-"+subject_id,"SPM.mat"),
                                voi_file=op.join(data_folder, "nipype",model, "1stLevel/sub-"+subject_id,"VOI_"+roi_mask.split("/")[-1].split(".")[-2]+"_"+str(run_id)+'.mat'),
                                learning_type="surprise",
                                run=run_id,
                                ppi_name=roi_mask.split("/")[-1].split(".")[-2]+"_"+str(run_id),
                                model=model), name='ppi_node')          

                # Create a workflow
                workflow = Workflow(name='ppi_workflow', base_dir=output_path)
                workflow.add_nodes([PPI_node])
                try:
                # Run the workflow
                    workflow.run() 
                except:
                    continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("roi_mask", type=str)
    parser.add_argument("--model", type=str, default="model2")
    parser.add_argument("--data_folder", type=str, default="/shares/zne.uzh/multlearn")
    parser.add_argument("--mlab_path", type=str, default="/usr/local/MATLAB/R2022b/bin/matlab")
    parser.add_argument("--spm_path", type=str, default="~/spm12")

    args = parser.parse_args()

    main(args.roi_mask, model = args.model, data_folder = args.data_folder, mlab_path = args.mlab_path, spm_path = args.spm_path)
