from nilearn import plotting
from os.path import join as opj
import json
import os
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

MatlabCommand.set_default_paths("~/spm12")
MatlabCommand.set_default_matlab_cmd("/usr/local/MATLAB/R2022b/bin/matlab")
fs_dir = "/mnt/d/data/ds-mlearn/derivatives/freesurfer"
FSCommand.set_default_subjects_dir(fs_dir)


class SnpmOneSampleTTestInputSpec(BaseInterfaceInputSpec):
    destination = traits.Directory(
        exists=True, desc="Output directory for SnPM results"
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


class SnpmOneSampleTTestOutputSpec(TraitedSpec):
    results = traits.List(traits.File, desc="List of SnPM result files")


class SnpmOneSampleTTest(BaseInterface):
    input_spec = SnpmOneSampleTTestInputSpec
    output_spec = SnpmOneSampleTTestOutputSpec
    _jobtype = "tools.snpm.des"
    _jobname = "OneSampT"

    def _run_interface(self, runtime):
        # Construct the SnPM command based on input specifications
        snpm_command = "function SnPM_script()\n"
        snpm_command += "addpath('~/spm12');\n"
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
            snpm_command += f"matlabbatch{{1}}.spm.tools.snpm.des.OneSampT.masking.em = {self.inputs.explicit_mask};\n"
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

        # Write the SnPM command to a temporary script
        script_path = "/mnt/d/multlearn-sns/SPM/nipype/SnPM_script.m"
        with open(script_path, "w") as script_file:
            script_file.write(snpm_command)

        # Run the SnPM command using subprocess
        subprocess.run(
            f"matlab -nodisplay -nosplash -r \"run('{script_path}'); exit;\"",
            shell=True,
            check=True,
        )

        return runtime

    def _list_outputs(self):
        return self._results

def get_subject_info(subject):
    from glob import glob
    import numpy as np
    import pandas as pd
    from scipy import io, stats
    from nipype.interfaces.base import Bunch

    subject_info = []
    rpe_path = (
        f"/mnt/d/multlearn-sns/Modelling/Fitting/bestFittingVals/sub-{subject}/rpe*.mat"
    )
    fn = glob(rpe_path)

    assert len(fn) == 1
    fn = fn[0]

    rpe_data = np.nan_to_num(
        stats.zscore(io.loadmat(fn)["rpe"], nan_policy="omit", axis=1)
    )

    rpe = (
        pd.DataFrame(
            rpe_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("rpe")
    )

    surprise_path = (
        f"/mnt/d/multlearn-sns/Modelling/Fitting/bestFittingVals/sub-{subject}/spe*.mat"
    )
    fn2 = glob(surprise_path)
    assert len(fn2) == 1
    fn2 = fn2[0]
    surprise_data = np.nan_to_num(
        stats.zscore(io.loadmat(fn2)["spe"], nan_policy="omit", axis=1)
    )
    surprise = (
        pd.DataFrame(
            surprise_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("spe")
    )

    functional_runs = []

    for run in range(1, 7):
        onsets = []
        durations = []
        conditions = []

        events_file = pd.read_csv(
            f"/mnt/d/data/ds-mlearn/derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-learn_run-{run}_events.tsv",
            delimiter="\t",
        )
        events_file_sorted = events_file.sort_values(by=["onset"])
        events_file_sorted["trial_nr"] = events_file_sorted["trial_nr"].ffill()
        events_file_sorted["runType"] = events_file_sorted["runType"].ffill()
        run_type = events_file_sorted["runType"][0]

        confounds = pd.read_csv(
            f"/mnt/d/data/ds-mlearn/derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-learn_run-{run}_desc-confounds_timeseries.tsv",
            delimiter="\t",
        )

        confounds = confounds.loc[
            :,
            [
                "trans_x",
                "trans_y",
                "trans_z",
                "rot_x",
                "rot_y",
                "rot_z",
                "a_comp_cor_00",
                "a_comp_cor_01",
                "a_comp_cor_02",
                "a_comp_cor_03",
                "a_comp_cor_04",
            ],
        ]

        physio_path = f"/mnt/d/data/ds-mlearn/derivatives/fmriprep/sub-{subject}/beh/physio/RegPhysio_sub-{subject}_run_{run}.mat"
        fn3 = glob(physio_path)
        assert len(fn3) == 1
        fn3 = fn3[0]

        physio = io.loadmat(fn3, simplify_cells=True)["physio"]["model"]
        physio = pd.DataFrame(
            data=physio["R"],
            columns=physio["R_column_names"],
        )

        regressors = pd.concat([confounds, physio], axis=1)
        regressor_names = regressors.columns.values.tolist()

        for group in events_file_sorted.groupby("trial_type"):
            conditions.append(str(group[0].capitalize() + run_type.capitalize()))
            onsets.append(group[1]["onset"].tolist())
            durations.append(group[1]["duration"].tolist())
        run_rpe = rpe.xs(run)
        run_surprise = surprise.xs(run)
        pmod = [
            Bunch(name=["surprise"], param=[run_surprise.values.tolist()], poly=[1]),
            Bunch(name=["rpe"], param=[run_rpe.values.tolist()], poly=[1]),
        ]

        subject_info.insert(
            run - 1,
            Bunch(
                conditions=conditions,
                onsets=onsets,
                durations=durations,
                pmod=pmod,
                tmod=None,
                orth=["No"] * len(conditions),
                regressors=regressors.values.T.tolist(),
                regressor_names=regressor_names,
            ),
        )

        functional_run = glob(
            f"/mnt/d/data/ds-mlearn/derivatives/fmriprep/sub-{subject}/func/s6.sub-{subject}_task-learn_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii"
        )[0]
        functional_runs.append(functional_run)

    return subject_info, functional_runs


def get_contrasts(subject_info):
    from nipype.interfaces.spm import EstimateContrast
    import os

    condition_names = [
        "ChoiceAudio",
        "ChoiceTactile",
        "FeedbackAudio",
        "FeedbackTactile",
        "ChoiceAudioxsurprise^1",
        "ChoiceTactilexsurprise^1",
        "FeedbackAudioxrpe^1",
        "FeedbackTactilexrpe^1",
    ]

    con01 = ["rpe", "T", condition_names[6:], [1 / 6.0, 1 / 6.0]]
    con02 = ["rpe_audio", "T", [condition_names[6]], [1 / 3.0]]
    con03 = ["rpe_tactile", "T", [condition_names[7]], [1 / 3.0]]
    con04 = [
        "rpe_audio < rpe_tactile",
        "T",
        [condition_names[6], condition_names[7]],
        [-1 / 3.0, 1 / 3.0],
    ]
    con05 = [
        "rpe_tactile < rpe_audio",
        "T",
        [condition_names[6], condition_names[7]],
        [1 / 3.0, -1 / 3.0],
    ]

    con06 = ["surprise", "T", condition_names[4:6], [1 / 6.0, 1 / 6.0]]
    con07 = ["surprise_audio", "T", [condition_names[4]], [1 / 3.0]]
    con08 = ["surprise_tactile", "T", [condition_names[5]], [1 / 3.0]]
    con09 = [
        "surprise_audio < surprise_tactile",
        "T",
        [condition_names[4], condition_names[5]],
        [-1 / 3.0, 1 / 3.0],
    ]
    con10 = [
        "surprise_tactile < surprise_audio",
        "T",
        [condition_names[4], condition_names[5]],
        [1 / 3.0, -1 / 3.0],
    ]

    con11 = [
        "rpe < surprise",
        "T",
        condition_names[4:],
        [-1 / 6.0, 1 / 6.0, -1 / 6.0, 1 / 6.0, -1 / 6.0, 1 / 6.0],
    ]
    con12 = [
        "surprise < rpe",
        "T",
        condition_names[4:],
        [1 / 6.0, -1 / 6.0, 1 / 6.0, -1 / 6.0, 1 / 6.0, -1 / 6.0],
    ]
    con13 = [
        "rpe_audio < surprise_audio",
        "T",
        [condition_names[4], condition_names[6]],
        [1 / 3.0, -1 / 3.0],
    ]
    con14 = [
        "surprise_audio < rpe_audio",
        "T",
        [condition_names[4], condition_names[6]],
        [-1 / 3.0, 1 / 3.0],
    ]
    con15 = [
        "rpe_tactile < surprise_tactile",
        "T",
        [condition_names[5], condition_names[7]],
        [1 / 3.0, -1 / 3.0],
    ]
    con16 = [
        "surprise_tactile < rpe_tactile",
        "T",
        [condition_names[5], condition_names[7]],
        [-1 / 3.0, 1 / 3.0],
    ]

    con17 = [
        "pmods",
        "T",
        condition_names[4:8],
        [1 / 12.0, 1 / 12.0, 1 / 12.0, 1 / 12.0],
    ]
    con18 = [
        "pmods_audio",
        "T",
        [condition_names[4], condition_names[6]],
        [1 / 6.0, 1 / 6.0],
    ]
    con19 = [
        "pmods_tactile",
        "T",
        [condition_names[5], condition_names[7]],
        [1 / 6.0, 1 / 6.0],
    ]
    con20 = [
        "pmods_audio-pmods_tactile",
        "T",
        condition_names[4:8],
        [-1 / 6.0, 1 / 6.0, -1 / 6.0, 1 / 6.0],
    ]
    con21 = [
        "pmods_tactile-pmods_audio",
        "T",
        condition_names[4:8],
        [1 / 6.0, -1 / 6.0, 1 / 6.0, -1 / 6.0],
    ]

    con22 = ["feedback", "T", condition_names[2:4], [1 / 6.0, 1 / 6.0]]
    con23 = ["choice", "T", condition_names[0:2], [1 / 6.0, 1 / 6.0]]
    con24 = [
        "feedback < choice",
        "T",
        condition_names[0:4],
        [1 / 6.0, 1 / 6.0, -1 / 6.0, -1 / 6.0],
    ]
    con25 = [
        "choice < feedback",
        "T",
        condition_names[0:4],
        [-1 / 6.0, -1 / 6.0, 1 / 6.0, 1 / 6.0],
    ]

    con_list = [
        con01,
        con02,
        con03,
        con04,
        con05,
        con06,
        con07,
        con08,
        con09,
        con10,
        con11,
        con12,
        con13,
        con14,
        con15,
        con16,
        con17,
        con18,
        con19,
        con20,
        con21,
        con22,
        con23,
        con24,
        con25,
    ]

    return con_list


def main(BIDS="/mnt/d/data/ds-mlearn/", Nslices=40, refSlice=20):
    layout = BIDSLayout(BIDS, derivatives=True)
    # list of subject identifiers
    subject_list = layout.get_subjects()
    subject_list = [sub for sub in subject_list if int(sub) not in [8, 13, 16, 31, 32, 44]]
    with open(
    "/mnt/d/data/ds-mlearn/derivatives/fmriprep/sub-01/func/sub-01_task-learn_run-1_space-T1w_desc-preproc_bold.json",
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

    getsubjectinfo = Node(
        Function(
            input_names=["subject"],
            output_names=["subject_info", "functional_runs"],
            function=get_subject_info,
        ),
        name="getsubjectinfo",
    )
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
            flags={"mthresh": 0.8, "globalnorm": "None"},
            mask_image="/mnt/d/multlearn-sns/SPM/mask_ICV.nii",
            volterra_expansion_order=1,
        ),
        name="level1design",
    )
    level1estimate = Node(
        EstimateModel(estimation_method={"Classical": 1}, write_residuals=False),
        name="level1estimate",
    )
    level1conest = Node(EstimateContrast(), name="level1conest")

    base_dir = "/mnt/d/multlearn-sns/SPM/nipype"
    output_dir = 'model1'
    working_dir = 'workingdir'
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("BIDS", type=str, default="/mnt/d/data/ds-mlearn/")
    parser.add_argument("--Nslices", type=int, default=40)
    parser.add_argument("--refSlice", type=int, default=20)

    args = parser.parse_args()

    main(args.BIDS, Nslices=args.Nslices, refSlice=args.refSlice)
