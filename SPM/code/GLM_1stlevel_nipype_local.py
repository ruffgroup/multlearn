from nilearn import plotting
from os.path import join as opj
import os.path as op
import json
import os
from nipype.interfaces.base import Bunch
import nipype.interfaces.spm as spm
from nipype.interfaces.spm import (
    Level1Design,
    EstimateModel,
    EstimateContrast,
    SPMCommand,
    Info,
    model,
)
from nipype.interfaces.matlab import MatlabCommand
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
import nipype_helpers


def main(data_folder="/mnt/d/data/", base_dir="/mnt/d/multlearn-sns/SPM", mask="/mnt/d/multlearn-sns/SPM/mask_ICV.nii", ppi_mask=None, model="model1", bestfitting_path='/mnt/d/multlearn-sns/Modelling/Fitting/bestFittingVals/', Nslices=40, refSlice=20, mlab_path="/usr/local/MATLAB/R2022b/bin/matlab", spm_path="~/spm12"):
    MatlabCommand.set_default_paths(spm_path)
    MatlabCommand.set_default_matlab_cmd(mlab_path)
    FSCommand.set_default_subjects_dir(opj(data_folder, "ds-mlearn/derivatives/freesurfer/"))
    subject_list = range(1,65)
    subject_list = [str(sub).zfill(2) for sub in subject_list if not in [8, 13, 16, 31, 32, 44]]


    with open((opj(data_folder,
    "ds-mlearn/derivatives/fmriprep/sub-01/func/sub-01_task-learn_run-1_space-T1w_desc-preproc_bold.json")),
    "rt",
    ) as fp:
        task_info = json.load(fp)

    TR = task_info["RepetitionTime"]

    infosource = Node(
        IdentityInterface(
            fields=[
                "subject_id",
            ],
        ),
        name="infosource",
    )
    infosource.iterables = [
        ("subject_id", subject_list),
    ]
    if model == "ROI" or model == "neurosynth_ROI":
        get_subject_info = getattr(nipype_helpers, 'get_subject_info_model2')
    elif model == "PPI" or model == "neurosynth_PPI":
        get_subject_info = getattr(nipype_helpers, 'get_subject_info_PPI')
    else:
        get_subject_info = getattr(nipype_helpers, 'get_subject_info_'+model) 

    if model == "PPI" or model == "neurosynth_PPI":
        getsubjectinfo = Node(
            Function(
                input_names=["subject", "data_folder", "roi_mask", "bestfitting_path"],
                output_names=["subject_info", "functional_runs"],
                function=get_subject_info,
            ),
            name="getsubjectinfo",
        )
        getsubjectinfo.inputs.roi_mask = ppi_mask
    else:
        getsubjectinfo = Node(
            Function(
                input_names=["subject", "data_folder", "bestfitting_path"],
                output_names=["subject_info", "functional_runs"],
                function=get_subject_info,
            ),
            name="getsubjectinfo",
        )
    getsubjectinfo.inputs.data_folder = data_folder
    getsubjectinfo.inputs.bestfitting_path = bestfitting_path
    
    if model == "neurosynth_ROI":
        get_contrasts = getattr(nipype_helpers, 'get_contrasts_ROI')
    elif model == "neurosynth_PPI":
        get_contrasts = getattr(nipype_helpers, 'get_contrasts_PPI')
    else:
        get_contrasts = getattr(nipype_helpers, 'get_contrasts_'+model)
    getcontrasts = Node(
        Function(
            input_names=["subject_info"],
            output_names=["contrasts"],
            function=get_contrasts,
        ),
        name="getcontrasts",
    )
    modelspec = Node(
        SpecifySPMModel(
            concatenate_runs=False,
            input_units="secs",
            output_units="secs",
            time_repetition=TR,
            high_pass_filter_cutoff=128,
        ),
        name="modelspec",
    )

    level1design = Node(
        Level1Design(
            bases={"hrf": {"derivs": [0, 0]}},
            timing_units="secs",
            interscan_interval=TR,
            model_serial_correlations="AR(1)",
            microtime_resolution=Nslices,
            microtime_onset=refSlice,
            flags={"mthresh": 0.8, "global": "None"},
            mask_image=mask,
            volterra_expansion_order=1,
        ),
        name="level1design",
    )
    level1estimate = Node(
        EstimateModel(estimation_method={"Classical": 1}, write_residuals=False),
        name="level1estimate",
    )
    level1conest = Node(EstimateContrast(), name="level1conest")

    if model == "ROI":
        output_dir = ''.join(mask.split("/")[-1].split("_")[2])+"/"+''.join(mask.split("/")[-1].split("_")[:2])+'_'+mask.split("/")[-1].split("_")[-1].split(".")[0]
        working_dir = "workingdir_"+''.join(mask.split("/")[-1].split("_")[:3])+"_"+mask.split("_")[-1].split(".")[0]
    elif model == "PPI":
        output_dir = op.join(ppi_mask.split("/")[-3], "PPI",''.join(ppi_mask.split("/")[-1].split("_")[2])+"/"+''.join(ppi_mask.split("/")[-1].split("_")[:2])+'_'+ppi_mask.split("/")[-1].split("_")[-1].split(".")[0])
        working_dir = op.join(ppi_mask.split("/")[-3], "PPI", "workingdir_"+''.join(ppi_mask.split("/")[-1].split("_")[:3])+"_"+ppi_mask.split("_")[-1].split(".")[0])
    elif model == "neurosynth_ROI":
        output_dir = op.join("neurosynth",''.join(mask.split(".nii")[0].split("/")[-1].split("_")[:2]))
        working_dir = op.join("neurosynth",'workingdir_'+''.join(mask.split(".nii")[0].split("/")[-1].split("_")[:2]))
    else:
        output_dir = model
        working_dir = 'workingdir_'+model
    
    base_dir=opj(base_dir,"nipype")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    datasink = Node(DataSink(base_directory=base_dir,
                         container=output_dir),
                name="datasink")
    substitutions = [('_subject_id_', 'sub-')]
    subjFolders = [('_sub-%s' % (sub), 'sub-%s' % (sub))
                for sub in subject_list]
    substitutions.extend(subjFolders)
    datasink.inputs.substitutions = substitutions
    
    first_level_wf = Workflow(name="first_level_wf", base_dir=os.path.join(base_dir, working_dir))


    # Connect the nodes
    first_level_wf.connect(
        [
            (infosource, getsubjectinfo, [("subject_id", "subject")]),
            (
                getsubjectinfo,
                modelspec,
                [
                    ("subject_info", "subject_info"),
                    ("functional_runs", "functional_runs"),
                ],
            ),
            (modelspec, level1design, [("session_info", "session_info")]),
            (
                level1design,
                level1estimate,
                [("spm_mat_file", "spm_mat_file")],
            ),
            (getsubjectinfo, getcontrasts, [("subject_info", "subject_info")]),
              # Connect level1design to level1estimate
            (
                getcontrasts,
                level1conest,
                [("contrasts", "contrasts")],
            ),  # Connect the contrasts to EstimateContrast
            (
                level1estimate,
                level1conest,
                [
                    ("spm_mat_file", "spm_mat_file"),
                    ("beta_images", "beta_images"),
                    ("residual_image", "residual_image"),
                ],
            ),
            (level1conest, datasink, [('spm_mat_file', '1stLevel.@spm_mat'),
                                              ('spmT_images', '1stLevel.@T'),
                                              ('con_images', '1stLevel.@con'),
                                              ]),
        ]
    )

    first_level_wf.config["logging"] = {
        "workflow_level": "DEBUG",
        "filemanip_level": "DEBUG",
        "interface_level": "DEBUG",
        "log_to_file": "True",
        "log_directory": "/output/log_folder",
    }

    first_level_wf.run('MultiProc', plugin_args={'n_procs': 2})


if __name__ == "__main__":
    
    data_folder="/mnt/d/data/"
    base_dir="/mnt/d/multlearn-sns/SPM/nipype"
    mask="/mnt/d/multlearn-sns/SPM/mask_ICV.nii"
    ppi_mask=None
    model="model2"
    bestfitting_path='/mnt/d/data/fittedParametersRecoveredModels/bestFittingVals'
    mlab_path="/usr/local/MATLAB/R2022b/bin/matlab"
    spm_path="~/spm12"
    Nslices=40
    refSlice=20

    main(data_folder, base_dir=base_dir, mask=mask, ppi_mask=ppi_mask, model=model, bestfitting_path=bestfitting_path, mlab_path=mlab_path, spm_path=spm_path, Nslices=Nslices, refSlice=refSlice)
