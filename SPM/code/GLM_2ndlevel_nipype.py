from nilearn import plotting
from os.path import join as opj
import json
import os
import re
from nipype.interfaces.base import (
    Bunch,
    BaseInterface,
    BaseInterfaceInputSpec,
    traits,
    File,
    TraitedSpec,
    InputMultiPath,
    )
import nipype.interfaces.spm as spm
from nipype.interfaces.spm import (
    Level1Design,
    EstimateModel,
    EstimateContrast,
    SPMCommand,
    Info,
    model,
)
import subprocess
from nipype.interfaces.matlab import MatlabCommand, MatlabInputSpec
from nipype.interfaces.freesurfer import FSCommand
from nipype.algorithms.modelgen import SpecifySPMModel, SpecifyModel
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype import Workflow, Node
from bids.layout import BIDSLayout
from glob import glob
from scipy import io, stats
from itertools import chain
import pandas as pd
import numpy as np
import pytest as pt
import nibabel as nb
import nipype
import argparse
import nipype.interfaces.matlab as matlab
from pathlib import Path


class SnpmOneSampleTTestInputSpec(BaseInterfaceInputSpec):
    destination = traits.Directory(
        exists=True, desc="Output directory for SnPM results"
    )

    base_dir = traits.Directory(
        exists=True, desc="nipype base directory"
    )

    spm_path = traits.Directory(
        exists=True, desc="spm directory"
    )
    inference = traits.Any(
        traits.Any, field="inference", desc="cluster or voxel-wise inference"
    )

    contrasts = traits.List(traits.Any, desc="List of contrast files")
    covariates = InputMultiPath(
        traits.Dict(key_trait=traits.Enum("c", "cname")),
        field="cov",
        desc="Covariate dictionary {c, cname}",
    )
    n_perms = traits.Int(
        #usedefault=True,
        #default_value=5000,
        field="nPerm",
        desc="Number of permutations",
    )
    var_smoothing = traits.List(
        #default_value=[0, 0, 0],
        #usedefault=True,
        field="vFWHM",
        desc="Smoothing kernel",
    )
    memory_usage = traits.Bool(
        #default_value=False,
        #usedefault=True,
        field="bVolm",
        desc="Memory usage (False = low, True = high)",
    )
    cluster_inference_none = traits.Int(
        0,
        xor=["cluster_inference_later", "cluster_inference_fast"],
        field="ST.ST_none",
        desc="No cluster inference 0",
    )
    cluster_inference_later = traits.Int(
        -1,
        xor=["cluster_inference_none", "cluster_inference_fast"],
        field="ST.ST_later",
        desc="Cluster inference slow -1",
    )
    cluster_inference_fast = traits.Float(
        xor=["cluster_inference_none", "cluster_inference_later"],
        field="ST.ST_U",
        desc="Cluster inference fast t-value",
    )
    masking_none = traits.Int(
        1,
        field="masking.tm.tm_none",
        xor=["thresh_mask_abs", "thresh_mask_rel"],
        desc="No masking 1",
    )
    thresh_mask_abs = traits.Float(
        field="masking.tm.tma.athresh",
        xor=["masking_none", "thresh_mask_rel"],
        desc="Absolute threshold masking (in voxels)",
    )
    thresh_mask_rel = traits.Float(
        field="masking.tm.tmr.rthresh",
        xor=["masking_none", "thresh_mask_abs"],
        desc="Relative threshold masking (proportion of global value)",
    )
    implicit_mask = traits.Enum(
        0,
        1,
        #usedefault=True,
        field="masking.im",
        desc="Implicit masking (0 = No, 1 = Yes)",
    )
    explicit_mask = traits.Any(
        traits.Any, field="masking.em", desc="Explicit mask files"
    )

    global_calc_omit = traits.Int(
        1,
        field="globalc.g_omit",
        xor=["global_calc_user", "global_calc_mean"],
        desc="Omit global calculation 1",
    )
    global_calc_user = traits.List(
        field="globalc.g_user",
        xor=["global_calc_omit", "global_calc_mean"],
        desc="User-defined global values",
    )
    global_calc_mean = traits.Int(
        1,
        field="globalc.g_mean",
        xor=["global_calc_omit", "global_calc_user"],
        desc="Mean global calculation 1",
    )
    no_grand_mean_scaling = traits.Int(
        1,
        field="globalm.gmsca.gmsca_no",
        xor=["grand_mean_scaling"],
        desc="No grand mean scaling 1",
    )
    grand_mean_scaling = traits.List(
        #default_value=[50],
        #usedefault=True,
        field="globalm.gmsca.gmsca_yes.gmscv",
        xor=["no_grand_mean_scaling"],
        desc="Grand mean scaling",
    )
    global_normalization = traits.Enum(
        1,
        2,
        3,
        mandatory=True,
        field="globalm.glonorm",
        desc="Global normalization (1 = None, 2 = Proportional, 3 = ANCOVA)",
    )


class SnpmOneSampleTTest(BaseInterface):
    input_spec = SnpmOneSampleTTestInputSpec
    _jobtype = "tools.snpm.des"
    _jobname = "OneSampT"

    def _run_interface(self, runtime):
        # Construct the SnPM command based on input specifications
        snpm_command = "function SnPM_2ndlevel_script()\n"
        snpm_command += f"addpath('{self.inputs.spm_path}');\n"
        snpm_command += "spm_jobman('initcfg');\n"
        snpm_command += "matlabbatch{1}.spm.tools.snpm.des.OneSampT.DesignName = 'MultiSub: One Sample T test on diffs/contrasts';\n"
        snpm_command += "matlabbatch{1}.spm.tools.snpm.des.OneSampT.DesignFile = 'snpm_bch_ui_OneSampT';\n"
        snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.des.OneSampT.dir = cellstr('{self.inputs.destination}');\n"
        snpm_command += "matlabbatch{1}.spm.tools.snpm.des.OneSampT.P = {...\n"

        # Add contrast files to the command
        for contrast in self.inputs.contrasts:
            snpm_command += f"'{contrast}';...\n"

        snpm_command += "};\n"
        snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.des.OneSampT.nPerm = {self.inputs.n_perms};\n"
        snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.des.OneSampT.vFWHM = {self.inputs.var_smoothing};\n"
        snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.des.OneSampT.bVolm = {int(self.inputs.memory_usage)};\n"
        if self.inputs.cluster_inference_none:
            snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.des.OneSampT.ST.ST_none = {self.inputs.cluster_inference_none};\n"
        elif self.inputs.cluster_inference_later:
            snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.des.OneSampT.ST.ST_later = {self.inputs.cluster_inference_later};\n"
        elif self.inputs.cluster_inference_fast:
            snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.des.OneSampT.ST.ST_U = {self.inputs.cluster_inference_fast};\n"
        if self.inputs.masking_none:
            snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.des.OneSampT.masking.tm.tm_none = {self.inputs.masking_none};\n"
        elif self.inputs.thresh_mask_abs:
            snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.des.OneSampT.masking.tm.tma.athresh = {self.inputs.thresh_mask_abs};\n"
        elif self.inputs.thresh_mask_rel:
            snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.des.OneSampT.masking.tm.tmr.rthresh = {self.inputs.thresh_mask_rel};\n"
        snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.des.OneSampT.masking.im = {self.inputs.implicit_mask};\n"
        if self.inputs.explicit_mask:
            snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.des.OneSampT.masking.em = cellstr('{self.inputs.explicit_mask}');\n"
        else:
            snpm_command += "matlabbatch{1}.spm.tools.snpm.des.OneSampT.masking.em = cellstr('');\n"
        if self.inputs.global_calc_omit:
            snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.des.OneSampT.globalc.g_omit = {self.inputs.global_calc_omit};\n"
        elif self.inputs.global_calc_user:
            snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.des.OneSampT.globalc.g_user = {self.inputs.global_calc_user};\n"
        elif self.inputs.global_calc_mean:
            snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.des.OneSampT.globalc.g_mean = {self.inputs.global_calc_mean};\n"
        if self.inputs.no_grand_mean_scaling:
            snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.des.OneSampT.globalm.gmsca.gmsca_no = {self.inputs.no_grand_mean_scaling};\n"
        elif self.inputs.grand_mean_scaling:
            snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.des.OneSampT.globalm.gmsca.gmsca_yes.gmscv = {self.inputs.grand_mean_scaling};\n"
        snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.des.OneSampT.globalm.glonorm = {self.inputs.global_normalization};\n"

        snpm_command += "matlabbatch{2}.spm.tools.snpm.cp.snpmcfg(1) = cfg_dep('MultiSub: One Sample T test on diffs/contrasts: SnPMcfg.mat configuration file', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','SnPMcfg'));\n"
        snpm_command += "spm('defaults', 'fMRI');\n"
        snpm_command += "spm_jobman('run', matlabbatch);\n"
        snpm_command += "clear matlabbatch\n"
        snpm_command += "end"

        
        script_path = opj(runtime.cwd, self.inputs.inference+"_SnPM_2ndlevel_script.m")
        with open(script_path, "w") as script_file:
            script_file.write(snpm_command)

        log_file= os.path.join(runtime.cwd, 'matlab.log')
        # Run the SnPM command using subprocess
        subprocess.run(
            f"matlab -nodisplay -nosplash -r \"run('{script_path}'); exit;\"",
            shell=True,
            check=True,
        )
        # mlab = matlab.MatlabCommand(script=script_path, paths=[self.inputs.spm_path], logfile=log_file)
        # result = mlab.run()

        return runtime
    
class SnpmInferenceInputSpec(BaseInterfaceInputSpec):
    snpm_mat_file = File(
        exists=True,
        field="SnPMmat",
        mandatory=True,
        desc="Absolute path to SnPM.mat",
    )

    inference = traits.Any(
        traits.Any, field="inference", desc="cluster or voxel-wise inference"
    )

    cluster_threshold = traits.Float(
        field="Thr.Clus.ClusSize.CFth",
        desc="Cluster-forming threshold",
    )
    fwe_threshold = traits.Float(
        0.05,
        field="Thr.Clus.ClusSize.ClusSig.FWEthC",
        usedefault=True,
        desc="Family-wise error threshold",
    )
    voxel_fwe_threshold = traits.Float(
        0.05,
        field="Thr.Vox.VoxSig.FWEth",
        usedefault=True,
        desc="Family-wise error threshold",
    )
    t_sign = traits.Int(
        1,
        field="Tsign",
        desc="T-value sign (1 for positive, -1 for negative)",
    )
    filtered_img_name = File(
        field="WriteFiltImg.name",
        xor=["no_filtered_img"],
        desc="Output file name for filtered image",
    )
    no_filtered_img = traits.Int(
        0,
        field="WriteFiltImg.WF_no",
        xor=["filtered_img_name"],
        desc="Do not write filtered image",
    )
    report_type = traits.Enum(
        "MIPtable",
        "FWEreport",
        "FDRreport",
        field="Report",
        desc="Report type (e.g., 'MIPtable')",
    )

    base_dir = traits.Directory(
        exists=True, desc="nipype base directory"
    )

    spm_path = traits.Directory(
        exists=True, desc="spm directory"
    )

    contrast = traits.Int(
        exists=True, desc="contrast nr"
    )


class SnpmInference(BaseInterface):
    input_spec = SnpmInferenceInputSpec
    _jobtype = "tools.snpm.inference"

    def _run_interface(self, runtime):
        # Construct the SnPM inference command based on input specifications
        if self.inputs.cluster_threshold:
            snpm_command = "function SnPM_inference_"+str(self.inputs.cluster_threshold).replace('.','_')+"_script()\n"
        elif self.inputs.voxel_fwe_threshold:
            snpm_command = "function SnPM_inference_"+str(self.inputs.voxel_fwe_threshold).replace('.','_')+"_script()\n"
        snpm_command += f"addpath('{self.inputs.spm_path}');\n"
        snpm_command += "spm_jobman('initcfg');\n"
        snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.inference.SnPMmat = cellstr('{self.inputs.snpm_mat_file}');\n"
        if self.inputs.cluster_threshold:
            snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.inference.Thr.Clus.ClusSize.CFth = {self.inputs.cluster_threshold};\n"
            snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.inference.Thr.Clus.ClusSize.ClusSig.FWEthC = {self.inputs.fwe_threshold};\n"
        elif self.inputs.voxel_fwe_threshold:
            snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.inference.Thr.Vox.VoxSig.FWEth = {self.inputs.voxel_fwe_threshold};\n"
        snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.inference.Tsign = {self.inputs.t_sign};\n"
        if self.inputs.filtered_img_name:
            snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.inference.WriteFiltImg.name = '{self.inputs.filtered_img_name}';\n"
        elif self.inputs.no_filtered_img:
            snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.inference.WriteFiltImg.WF_no = {self.inputs.no_filtered_img};\n"
        snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.inference.Report = '{self.inputs.report_type}';\n"
        snpm_command += "spm('defaults', 'fMRI');\n"
        snpm_command += "spm_jobman('run', matlabbatch);\n"
        if self.inputs.filtered_img_name:
            snpm_command += f"savefig('{self.inputs.filtered_img_name}.fig')\n"
        snpm_command += "clear matlabbatch\n"
        snpm_command += "end"

        if self.inputs.cluster_threshold:
            if self.inputs.t_sign == 1:
                script_path_inf = opj(self.inputs.base_dir,"2ndLevel",self.inputs.inference+"_SnPM_SecondLevel_con"+str(self.inputs.contrast), "SnPM_inference_"+str(self.inputs.cluster_threshold).replace('.','_')+"_pos_script.m")
            else:
                script_path_inf = opj(self.inputs.base_dir,"2ndLevel",self.inputs.inference+"_SnPM_SecondLevel_con"+str(self.inputs.contrast), "SnPM_inference_"+str(self.inputs.cluster_threshold).replace('.','_')+"_neg_script.m")
            with open(script_path_inf, "w") as script_file:
                script_file.write(snpm_command)
        elif self.inputs.voxel_fwe_threshold:
            if self.inputs.t_sign == 1:
                script_path_inf = opj(self.inputs.base_dir,"2ndLevel",self.inputs.inference+"_SnPM_SecondLevel_con"+str(self.inputs.contrast), "SnPM_inference_"+str(self.inputs.voxel_fwe_threshold).replace('.','_')+"_pos_script.m")
            else:
                script_path_inf = opj(self.inputs.base_dir,"2ndLevel",self.inputs.inference+"_SnPM_SecondLevel_con"+str(self.inputs.contrast), "SnPM_inference_"+str(self.inputs.voxel_fwe_threshold).replace('.','_')+"_neg_script.m")
            with open(script_path_inf, "w") as script_file:
                script_file.write(snpm_command)

        subprocess.run(
            f"matlab -nodisplay -nosplash -r \"run('{script_path_inf}'); exit;\"",
            shell=True,
            check=True,
        )

        return runtime


def main(contrast, tVal=[3.1], data_folder="/mnt/d/data/",model="fmri",inference="cluster", base_dir="/mnt/d/multlearn-sns/SPM/nipype/", mlab_path="/usr/local/MATLAB/R2022b/bin/matlab", spm_path="~/spm12",
         roi=None):
    MatlabCommand.set_default_paths(spm_path)
    MatlabCommand.set_default_matlab_cmd(mlab_path)
    # FSCommand.set_default_subjects_dir(opj(data_folder, "ds-mlearn/derivatives/freesurfer/"))
    subject_list = range(1,65)
    subject_list = [str(sub).zfill(2) for sub in subject_list if sub not in [8, 13, 16, 31, 32, 44]]
    SnPM_2nd = Node(SnpmOneSampleTTest(), name=f'SnPM_contrast{contrast}')
    sub_contrasts = list()
    for sub in subject_list:
        path_1stlevel = opj(base_dir,f'1stLevel/sub-{sub}/con_{contrast:04d}.nii')
        if os.path.exists(path_1stlevel):
            sub_contrasts.append(path_1stlevel)
        else:
            print(f"Did not find contrast: {path_1stlevel}")
    SnPM_2nd.inputs.contrasts = sub_contrasts
    destination = opj(base_dir,'2ndLevel',inference+'_SnPM_SecondLevel_con') + str(contrast)
    if inference == "cluster":
        SnPM_2nd.inputs.cluster_inference_later = -1
    elif inference == "voxel":
        SnPM_2nd.inputs.cluster_inference_none = 0
    SnPM_2nd.inputs.inference = inference

    if roi is not None:
        mask = Path(data_folder) / "nipype" / "model2" / "ROI" / f"{roi}.nii"
        assert(mask.exists()), f"Mask {roi} not found: {mask}"
        destination += f"/{roi}"
        SnPM_2nd.inputs.explicit_mask = str(mask)+",1"

    if not os.path.exists(destination):
        os.makedirs(destination)
    SnPM_2nd.inputs.destination = destination
    # SnPM_2nd.inputs.base_dir = base_dir
    SnPM_2nd.inputs.base_dir = destination
    SnPM_2nd.inputs.spm_path = spm_path
    SnPM_2nd.inputs.n_perms = 5000
    SnPM_2nd.inputs.var_smoothing = [0, 0, 0]
    SnPM_2nd.inputs.memory_usage = False
    SnPM_2nd.inputs.masking_none = 1
    SnPM_2nd.inputs.implicit_mask = 1      
    if model == "ROI":
        con_cluster = base_dir.split("/")[-2:]
        temp = re.compile("([a-zA-Z]+)([0-9]+)")
        con = temp.match(con_cluster[0]).groups()[1]
        cluster = temp.match(con_cluster[1]).groups()[1]
        label = con_cluster[-1].split("_")[-1]
        mask = glob("/shares/zne.uzh/multlearn/nipype/model2/ROI/"+"cluster_"+str(cluster)+"_con"+str(con)+"*"+str(label)+".nii")[0]
        SnPM_2nd.inputs.explicit_mask = str(mask)+",1"
    elif model == "neurosynth_ROI":
        mask = glob(base_dir.split("cluster")[0]+"cluster_"+base_dir.split("cluster")[-1]+".nii")[0]
        SnPM_2nd.inputs.explicit_mask = str(mask)+",1"




    SnPM_2nd.inputs.global_calc_omit = 1
    SnPM_2nd.inputs.no_grand_mean_scaling = 1
    SnPM_2nd.inputs.global_normalization = 1

    SnPM_2nd.run()
    if inference == "cluster":
        for i in range(len(tVal)):
            SnPM_inf = Node(SnpmInference(), name=f'SnPM_inf_contrast{contrast}_thr_{tVal[i]}_neg'.replace('.', '_'))
            SnPM_inf.inputs.inference = inference
            SnPM_inf.inputs.base_dir = base_dir
            SnPM_inf.inputs.spm_path = spm_path
            SnPM_inf.inputs.contrast = contrast
            # SnPM_inf.inputs.snpm_mat_file = opj(base_dir,'2ndLevel',inference+'_SnPM_SecondLevel_con') + str(contrast) + '/SnPM.mat'
            SnPM_inf.inputs.snpm_mat_file = opj(destination, 'SnPM.mat')
            SnPM_inf.inputs.cluster_threshold = tVal[i]
            SnPM_inf.inputs.fwe_threshold = 0.05

            # t_sign = -1
            SnPM_inf.inputs.t_sign = -1
            # SnPM_inf.inputs.filtered_img_name = opj(base_dir,'2ndLevel',inference+'_SnPM_SecondLevel_con') + str(contrast) + '/SnPM_filtered_t'+str(tVal[i]).replace('.', '_')+'_neg'
            SnPM_inf.inputs.filtered_img_name = opj(destination, 'SnPM_filtered_t'+str(tVal[i]).replace('.', '_')+'_neg')
            SnPM_inf.inputs.report_type = 'MIPtable'
            SnPM_inf.run()

            if not os.path.exists(opj(base_dir,'2ndLevel',inference+'_SnPM_SecondLevel_con')+ str(contrast) + '/SnPM_filtered_t'+str(tVal[i]).replace('.', '_')+'_neg.nii'):
                break

        for i in range(len(tVal)):
            SnPM_inf = Node(SnpmInference(), name=f'SnPM_inf_contrast{contrast}_thr_{tVal[i]}_pos'.replace('.', '_'))
            SnPM_inf.inputs.inference = inference
            SnPM_inf.inputs.base_dir = base_dir
            SnPM_inf.inputs.spm_path = spm_path
            SnPM_inf.inputs.contrast = contrast
            # SnPM_inf.inputs.snpm_mat_file = opj(base_dir,'2ndLevel',inference+'_SnPM_SecondLevel_con') + str(contrast) + '/SnPM.mat'
            SnPM_inf.inputs.snpm_mat_file = opj(destination, 'SnPM.mat')
            SnPM_inf.inputs.cluster_threshold = tVal[i]
            SnPM_inf.inputs.fwe_threshold = 0.05

            # t_sign = 1
            SnPM_inf.inputs.t_sign = 1
            # SnPM_inf.inputs.filtered_img_name = opj(base_dir,'2ndLevel',inference+'_SnPM_SecondLevel_con') + str(contrast) + '/SnPM_filtered_t'+str(tVal[i]).replace('.', '_')+'_pos'
            SnPM_inf.inputs.filtered_img_name = opj(destination, 'SnPM_filtered_t'+str(tVal[i]).replace('.', '_')+'_pos')
            SnPM_inf.inputs.report_type = 'MIPtable'
            SnPM_inf.run()

            if not os.path.exists(opj(base_dir,'2ndLevel',inference+'_SnPM_SecondLevel_con')+ str(contrast) + '/SnPM_filtered_t'+str(tVal[i]).replace('.', '_')+'_pos.nii'):
                break
    elif inference == "voxel":
        SnPM_inf = Node(SnpmInference(), name=f'SnPM_inf_contrast{contrast}_thr_{tVal[i]}_neg'.replace('.', '_'))
        SnPM_inf.inputs.inference = inference
        SnPM_inf.inputs.base_dir = base_dir
        SnPM_inf.inputs.spm_path = spm_path
        SnPM_inf.inputs.contrast = contrast
        # SnPM_inf.inputs.snpm_mat_file = opj(base_dir,'2ndLevel',inference+'_SnPM_SecondLevel_con') + str(contrast) + '/SnPM.mat'
        SnPM_inf.inputs.snpm_mat_file = opj(destination, 'SnPM.mat')
        SnPM_inf.inputs.voxel_fwe_threshold = 0.05

        SnPM_inf.inputs.t_sign = 1
        # SnPM_inf.inputs.filtered_img_name = opj(base_dir,'2ndLevel',inference+'_SnPM_SecondLevel_con') + str(contrast) + '/SnPM_filtered_pos'
        SnPM_inf.inputs.filtered_img_name = opj(destination, 'SnPM_filtered_pos')
        
        SnPM_inf.inputs.report_type = 'MIPtable'
        SnPM_inf.run()

        SnPM_inf = Node(SnpmInference(), name=f'SnPM_inf_contrast{contrast}_thr_{tVal[i]}_pos'.replace('.', '_'))

        SnPM_inf.inputs.inference = inference
        SnPM_inf.inputs.base_dir = base_dir
        SnPM_inf.inputs.spm_path = spm_path
        SnPM_inf.inputs.contrast = contrast
        # SnPM_inf.inputs.snpm_mat_file = opj(base_dir,'2ndLevel',inference+'_SnPM_SecondLevel_con') + str(contrast) + '/SnPM.mat'
        SnPM_inf.inputs.snpm_mat_file = opj(destination, 'SnPM.mat')
        SnPM_inf.inputs.voxel_fwe_threshold = 0.05
        
        SnPM_inf.inputs.t_sign = -1
        # SnPM_inf.inputs.filtered_img_name = opj(base_dir,'2ndLevel',inference+'_SnPM_SecondLevel_con') + str(contrast) + '/SnPM_filtered_neg'
        SnPM_inf.inputs.filtered_img_name = opj(destination, 'SnPM_filtered_neg')
        SnPM_inf.inputs.report_type = 'MIPtable'
        SnPM_inf.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("contrast", type=int, default=1)
    parser.add_argument("data_folder", type=str, default="/shares/zne.uzh/multlearn")
    parser.add_argument("--model", type=str, default="fmri")
    parser.add_argument("--inference", type=str, default="cluster")
    parser.add_argument("--base_dir", type=str, default="/mnt/d/multlearn-sns/SPM/nipype/")
    parser.add_argument("--mlab_path", type=str, default="/usr/local/MATLAB/R2022b/bin/matlab")
    parser.add_argument("--spm_path", type=str, default="~/spm12")
    parser.add_argument("--tVal", type=float, nargs='+', default=[3.1])
    parser.add_argument("--roi", type=str, default=None)

    args = parser.parse_args()

    main(contrast=args.contrast, model=args.model,inference=args.inference,data_folder=args.data_folder, base_dir=args.base_dir, mlab_path=args.mlab_path, spm_path=args.spm_path, tVal=args.tVal,
         roi=args.roi)
