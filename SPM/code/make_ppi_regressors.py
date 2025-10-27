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
from pathlib import Path

class SpmPPIInputSpec(BaseInterfaceInputSpec):
    spm_path = Directory(exists=True, desc='SPM directory', mandatory=True)
    spm_mat_file = File(exists=True, desc='Path to SPM.mat file', mandatory=True)
    voi_file = File(exists=True, desc='Path to VOI.mat file', mandatory=True)
    variable = traits.Str(desc='rpe or surprise', mandatory=True)
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
        variable = self.inputs.variable
        ppi_name = self.inputs.ppi_name
        run = self.inputs.run
        model = self.inputs.model

        assert(model == 'model7'), 'Only model 7 is implemented for now.'
        
        matlab_script = f"""
        addpath('{spm_path}');
        spm('defaults', 'FMRI');
        spm_jobman('initcfg');
        matlabbatch{{1}}.spm.stats.ppi.spmmat = {{'{spm_mat_file}'}};
        matlabbatch{{1}}.spm.stats.ppi.type.ppi.voi = {{'{voi_file}'}};
        if contains('{variable}', 'surprise') 
            matlabbatch{{1}}.spm.stats.ppi.type.ppi.u = [1 1 0
                1 2 1
                1 3 0
                2 1 0
                2 2 0];
        elseif contains('{variable}', 'urpe')
            matlabbatch{{1}}.spm.stats.ppi.type.ppi.u = [1 1 0
                1 2 0
                1 3 0
                2 1 0
                2 2 1];
        elseif contains('{variable}', 'rpe')
            matlabbatch{{1}}.spm.stats.ppi.type.ppi.u = [1 1 0
                1 2 0
                1 3 0
                2 1 0
                2 2 0
                2 3 1];
        elseif contains('{variable}', 'choice')
            matlabbatch{{1}}.spm.stats.ppi.type.ppi.u = [1 1 1
                1 2 0
                1 3 0
                2 1 0
                2 2 0];
        elseif contains('{variable}', 'feedback')
            matlabbatch{{1}}.spm.stats.ppi.type.ppi.u = [1 1 0
                1 2 0
                1 3 0
                2 1 1
                2 2 0];
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
    
# def main(roi_mask, model="model2", data_folder="/shares/zne.uzh/multlearn", mlab_path="/usr/local/MATLAB/R2022b/bin/matlab", spm_path="~/spm12"):

def main(subject, roi, model, variable, data_folder, mlab_path, spm_path, work_dir):
    MatlabCommand.set_default_paths(spm_path)
    MatlabCommand.set_default_matlab_cmd(mlab_path)


    data_folder = Path(data_folder)

    target_dir = data_folder / 'nipype' / model / 'PPI' / f'{roi}_{variable}' / f'sub-{subject:02d}'
    target_dir.mkdir(parents=True, exist_ok=True)

    ppi_nodes = []

    spm_folder = data_folder / 'nipype' / model / '1stLevel' / f'sub-{subject:02d}' 
    spm_mat_file = spm_folder / 'SPM.mat'

    if roi in ['A1_L', 'A1_R']:
        contrast = 17
    elif roi in ['S1_R']:
        contrast = 23
    elif roi in ['AG_S_L', 'AG_S_R', 'DLPFC_S_L']:
        contrast = 5
    elif roi in ['AG_RPE_L', 'V1_RPE_L']:
        contrast = 19
    else:
        raise NotImplementedError(f'Roi {roi} unknown')

    for run in range(1, 7):
        voi_file = spm_folder / f'VOI_{roi}_{contrast}_{run}_{run}.mat'
        node = Node(SpmPPI(spm_path=spm_path,
                    spm_mat_file=spm_mat_file,
                    voi_file=voi_file,
                    variable=variable,
                    run=run,
                    ppi_name=f'{roi}_{variable}_run_{run}', model=model), name=f'ppi_node_subject-{subject:02d}_roi-{roi}_variable-{variable}_run-{run}')

        ppi_nodes.append(node)
    

    # Create a workflow
    workflow = Workflow(name=f'ppi_node_subject-{subject:02d}_roi-{roi}_variable-{variable}', base_dir=work_dir)
    workflow.add_nodes(ppi_nodes)
    workflow.run(plugin='MultiProc', plugin_args={'n_procs': 8})

if __name__ == '__main__':

    parser = argparser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, type=int)
    parser.add_argument('--roi', default='A1_L', choices=['A1_L', 'A1_R', 'S1_R', 'V1_RPE_L', 'AG_S_L', 'AG_S_R', 'AG_RPE_L', 'DLPFC_S_L'])
    parser.add_argument('--model', default='model7', choices=['model7'])
    parser.add_argument(
        '--variable',
        choices=['urpe', 'rpe', 'surprise', 'choice', 'feedback'],
        default='rpe',
        help='Specify the variable to process. Choices are: urpe, surprise, choice, feedback.'
    )
    parser.add_argument("--data_folder", type=str, default="/shares/zne.uzh/multlearn")
    parser.add_argument("--mlab_path", type=str, default="/apps/opt/containers/bin/matlab/r2023b/matlab")
    parser.add_argument("--spm_path", type=str, default=op.join(os.environ['HOME'], 'spm12'))
    parser.add_argument("--work_dir", default=Path('/scratch') / os.environ['USER'] / 'working_dir')

    args = parser.parse_args()


    main(subject=args.subject, roi=args.roi, model=args.model, variable=args.variable, data_folder=args.data_folder, mlab_path=args.mlab_path, spm_path=args.spm_path,
         work_dir=args.work_dir)