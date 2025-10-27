# Different GLMs and contrasts for 1st level analysis

def get_subject_info_model1(subject, data_folder, bestfitting_path='/mnt/d/multlearn-sns/Modelling/Fitting/bestFittingVals/'):
    from glob import glob
    import numpy as np
    import pandas as pd
    from scipy import io, stats
    from nipype.interfaces.base import Bunch
    import os.path as op

    subject_info = []
    rpe_path = (
        op.join(bestfitting_path,f"sub-{subject}/rpeBest.mat")
    )
    fn = glob(rpe_path)
    print(fn)
    assert len(fn) == 1
    fn = fn[0]

    rpe_data = np.nan_to_num(
        io.loadmat(fn, simplify_cells=True)["rpe"]["rpe"]
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

    rpe["rpe"] = rpe["rpe"].groupby("run").transform(lambda x: stats.zscore(x - x.mean()))
    print(rpe)
    surprise_path = (
        op.join(bestfitting_path,f"sub-{subject}/spe*.mat")
    )
    fn2 = glob(surprise_path)
    
    assert len(fn2) == 1
    fn2 = fn2[0]
    surprise_data = np.nan_to_num(
        io.loadmat(fn2)["spe"]
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

    surprise["spe"] = surprise["spe"].groupby("run").transform(lambda x: stats.zscore(x - x.mean()))
    print(surprise)
    functional_runs = []

    for run in range(1, 7):
        onsets = []
        durations = []
        conditions = []

        events_file = pd.read_csv(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-learn_run-{run}_events.tsv"),
            delimiter="\t",
        )
        events_file_sorted = events_file.sort_values(by=["onset"])
        events_file_sorted["trial_nr"] = events_file_sorted["trial_nr"].ffill()
        events_file_sorted["runType"] = events_file_sorted["runType"].ffill()
        run_type = events_file_sorted["runType"][0]

        confounds = pd.read_csv(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-learn_run-{run}_desc-confounds_timeseries.tsv"),
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

        physio_path = op.join(data_folder, f"ds-mlearn/derivatives/fmriprep/sub-{subject}/beh/physio/RegPhysio_sub-{subject}_run_{run}.mat")
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

        functional_run = glob(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/s6.sub-{subject}_task-learn_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii"
        ))[0]
        functional_runs.append(functional_run)

    return subject_info, functional_runs


def get_subject_info_model2(subject, data_folder, bestfitting_path='/mnt/d/multlearn-sns/Modelling/Fitting/bestFittingVals/'):
    from glob import glob
    import numpy as np
    import pandas as pd
    from scipy import io, stats
    from nipype.interfaces.base import Bunch
    import os.path as op

    subject_info = []
    rpe_path = (
        op.join(bestfitting_path,f"sub-{subject}/rpeBest.mat")
    )
    fn = glob(rpe_path)
    print(fn)

    assert len(fn) == 1
    fn = fn[0]

    rpe_data = np.nan_to_num(
        io.loadmat(fn, simplify_cells=True)["rpe"]["rpe"]
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

    rpe["rpe"] = rpe["rpe"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    rpe["rpe"] = np.nan_to_num(rpe["rpe"])

    surprise_path = (
        op.join(bestfitting_path,f"sub-{subject}/spe*.npy")
    )
    fn2 = glob(surprise_path)
    print(fn2)
    assert len(fn2) == 1
    fn2 = fn2[0]
    surprise_data = np.load(fn2)
    
    surprise = (
        pd.DataFrame(
            surprise_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("spe")
    )
    surprise["spe"] = surprise["spe"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    surprise["spe"] = np.nan_to_num(surprise["spe"])

    V0_path = (
        op.join(bestfitting_path,f"sub-{subject}/V0*.npy")
    )
    fn3 = glob(V0_path)
    print(fn3)
    assert len(fn3) == 1
    fn3 = fn3[0]
    V0_data = np.load(fn3)
    V0 = (
        pd.DataFrame(
            V0_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("V0")
    )

    V1_path = (
        op.join(bestfitting_path,f"sub-{subject}/V1*.npy")
    )

    fn4 = glob(V1_path)
    print(fn4)
    assert len(fn4) == 1
    fn4 = fn4[0]
    V1_data = np.load(fn4)
    V1 = (
        pd.DataFrame(
            V1_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("V1")
    )

    fn5 = op.join(
            data_folder,
            "sourcedata",
            "behavior",
            f"modified_files",
            f"modified_participant{subject}_savedValues.csv",
            )
    stimulus_info = (
    pd.read_csv(
        fn5,
        usecols=["trialNumber", "runNumber", "reward", "action", "responseTime"],
    )
    .rename(columns={"trialNumber": "trial_nr", "runNumber": "run"})
    .set_index(["run", "trial_nr"])
    .sort_index()
    )

    V_df = pd.concat([V0, V1, stimulus_info], axis=1)
    V_df["action"] = V_df["action"].fillna(0.5)
    V_df["V"] = np.where(V_df["action"] == 0.0, V_df["V0"],V_df["V1"])
    V_df["V"] = V_df["V"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    V_df["V"] = np.where(V_df["action"] == 0.5, 0.0, V_df["V"])
    functional_runs = []

    for run in range(1, 7):
        onsets = []
        durations = []
        conditions = []

        events_file = pd.read_csv(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-learn_run-{run}_events.tsv"),
            delimiter="\t",
        )
        events_file_sorted = events_file.sort_values(by=["onset"])
        events_file_sorted["trial_nr"] = events_file_sorted["trial_nr"].ffill()
        events_file_sorted["runType"] = events_file_sorted["runType"].ffill()
        run_type = events_file_sorted["runType"][0]

        confounds = pd.read_csv(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-learn_run-{run}_desc-confounds_timeseries.tsv"),
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

        physio_path = op.join(data_folder, f"ds-mlearn/derivatives/fmriprep/sub-{subject}/beh/physio/RegPhysio_sub-{subject}_run_{run}.mat")
        fn6 = glob(physio_path)
        assert len(fn6) == 1
        fn6 = fn6[0]

        physio = io.loadmat(fn6, simplify_cells=True)["physio"]["model"]
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
        run_V = V_df["V"].xs(run)
    
        pmod = [
            Bunch(name=["surprise", "V"], param=[run_surprise.values.tolist(), run_V.values.tolist()], poly=[1, 1]),
            Bunch(name=["rpe"], param=[run_rpe.values.tolist()], poly=[1]),
        ]

        print(pmod)

        subject_info.insert(
            run - 1,
            Bunch(
                conditions=conditions,
                onsets=onsets,
                durations=durations,
                pmod=pmod,
                tmod=None,
                regressors=regressors.values.T.tolist(),
                regressor_names=regressor_names,
            ),
        )

        functional_run = glob(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/s6.sub-{subject}_task-learn_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii"
        ))[0]
        functional_runs.append(functional_run)

    print(subject_info)

    return subject_info, functional_runs #, V_df, rpe, surprise

def get_subject_info_model3(subject, data_folder, bestfitting_path='/mnt/d/multlearn-sns/Modelling/Fitting/bestFittingVals/'):
    from glob import glob
    import numpy as np
    import pandas as pd
    from scipy import io, stats
    from nipype.interfaces.base import Bunch
    import os.path as op

    subject_info = []
    rpe_path = (
        op.join(bestfitting_path,f"sub-{subject}/rpeBest.mat")
    )
    fn = glob(rpe_path)
    print(fn)

    assert len(fn) == 1
    fn = fn[0]

    rpe_data = np.nan_to_num(
        io.loadmat(fn, simplify_cells=True)["rpe"]["rpe"]
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

    rpe["rpe"] = rpe["rpe"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    rpe["rpe"] = np.nan_to_num(rpe["rpe"])

    surprise_path = (
        op.join(bestfitting_path,f"sub-{subject}/spe*.npy")
    )
    fn2 = glob(surprise_path)
    print(fn2)
    assert len(fn2) == 1
    fn2 = fn2[0]
    surprise_data = np.load(fn2)
    
    surprise = (
        pd.DataFrame(
            surprise_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("spe")
    )
    surprise["spe"] = surprise["spe"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    surprise["spe"] = np.nan_to_num(surprise["spe"])

    V0_path = (
        op.join(bestfitting_path,f"sub-{subject}/V0*.npy")
    )
    fn3 = glob(V0_path)
    print(fn3)
    assert len(fn3) == 1
    fn3 = fn3[0]
    V0_data = np.load(fn3)
    V0 = (
        pd.DataFrame(
            V0_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("V0")
    )

    V1_path = (
        op.join(bestfitting_path,f"sub-{subject}/V1*.npy")
    )

    fn4 = glob(V1_path)
    print(fn4)
    assert len(fn4) == 1
    fn4 = fn4[0]
    V1_data = np.load(fn4)
    V1 = (
        pd.DataFrame(
            V1_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("V1")
    )

    fn5 = op.join(
            data_folder,
            "sourcedata",
            "behavior",
            f"modified_files",
            f"modified_participant{subject}_savedValues.csv",
            )
    stimulus_info = (
    pd.read_csv(
        fn5,
        usecols=["trialNumber", "runNumber", "reward", "action", "responseTime"],
    )
    .rename(columns={"trialNumber": "trial_nr", "runNumber": "run"})
    .set_index(["run", "trial_nr"])
    .sort_index()
    )

    V_df = pd.concat([V0, V1, stimulus_info], axis=1)
    V_df["action"] = V_df["action"].fillna(0.5)
    V_df["V"] = np.where(V_df["action"] == 0.0, V_df["V0"],V_df["V1"])
    V_df["V"] = V_df["V"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    V_df["V"] = np.where(V_df["action"] == 0.5, 0.0, V_df["V"])
    functional_runs = []

    for run in range(1, 7):
        onsets = []
        durations = []
        conditions = []

        events_file = pd.read_csv(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-learn_run-{run}_events.tsv"),
            delimiter="\t",
        )
        events_file_sorted = events_file.sort_values(by=["onset"])
        events_file_sorted["trial_nr"] = events_file_sorted["trial_nr"].ffill()
        events_file_sorted["runType"] = events_file_sorted["runType"].ffill()
        events_file_sorted.loc[events_file_sorted.trial_type == "choice", "duration"] = 0
        run_type = events_file_sorted["runType"][0]

        confounds = pd.read_csv(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-learn_run-{run}_desc-confounds_timeseries.tsv"),
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

        physio_path = op.join(data_folder, f"ds-mlearn/derivatives/fmriprep/sub-{subject}/beh/physio/RegPhysio_sub-{subject}_run_{run}.mat")
        fn6 = glob(physio_path)
        assert len(fn6) == 1
        fn6 = fn6[0]

        physio = io.loadmat(fn6, simplify_cells=True)["physio"]["model"]
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
        run_V = V_df["V"].xs(run)
    
        pmod = [
            Bunch(name=["surprise", "V"], param=[run_surprise.values.tolist(), run_V.values.tolist()], poly=[1, 1]),
            Bunch(name=["rpe"], param=[run_rpe.values.tolist()], poly=[1]),
        ]

        print(pmod)

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

        functional_run = glob(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/s6.sub-{subject}_task-learn_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii"
        ))[0]
        functional_runs.append(functional_run)

    print(subject_info)

    return subject_info, functional_runs #, V_df, rpe, surprise

def get_subject_info_model4(subject, data_folder, bestfitting_path='/mnt/d/multlearn-sns/Modelling/Fitting/bestFittingVals/'):
    from glob import glob
    import numpy as np
    import pandas as pd
    from scipy import io, stats
    from nipype.interfaces.base import Bunch
    import os.path as op

    subject_info = []
    rpe_path = (
        op.join(bestfitting_path,f"sub-{subject}/rpeBest.mat")
    )
    fn = glob(rpe_path)
    print(fn)

    assert len(fn) == 1
    fn = fn[0]

    rpe_data = np.nan_to_num(
        io.loadmat(fn, simplify_cells=True)["rpe"]["rpe"]
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

    rpe["rpe"] = rpe["rpe"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    rpe["rpe"] = np.nan_to_num(rpe["rpe"])

    surprise_path = (
        op.join(bestfitting_path,f"sub-{subject}/spe*.npy")
    )
    fn2 = glob(surprise_path)
    print(fn2)
    assert len(fn2) == 1
    fn2 = fn2[0]
    surprise_data = np.load(fn2)
    
    surprise = (
        pd.DataFrame(
            surprise_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("spe")
    )
    surprise["spe"] = surprise["spe"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    surprise["spe"] = np.nan_to_num(surprise["spe"])

    V0_path = (
        op.join(bestfitting_path,f"sub-{subject}/V0*.npy")
    )
    fn3 = glob(V0_path)
    print(fn3)
    assert len(fn3) == 1
    fn3 = fn3[0]
    V0_data = np.load(fn3)
    V0 = (
        pd.DataFrame(
            V0_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("V0")
    )

    V1_path = (
        op.join(bestfitting_path,f"sub-{subject}/V1*.npy")
    )

    fn4 = glob(V1_path)
    print(fn4)
    assert len(fn4) == 1
    fn4 = fn4[0]
    V1_data = np.load(fn4)
    V1 = (
        pd.DataFrame(
            V1_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("V1")
    )

    fn5 = op.join(
            data_folder,
            "sourcedata",
            "behavior",
            f"modified_files",
            f"modified_participant{subject}_savedValues.csv",
            )
    stimulus_info = (
    pd.read_csv(
        fn5,
        usecols=["trialNumber", "runNumber", "reward", "action", "responseTime"],
    )
    .rename(columns={"trialNumber": "trial_nr", "runNumber": "run"})
    .set_index(["run", "trial_nr"])
    .sort_index()
    )

    V_df = pd.concat([V0, V1, stimulus_info], axis=1)
    V_df["action"] = V_df["action"].fillna(0.5)
    V_df["Vchosen"] = np.where(V_df["action"] == 0.0, V_df["V0"],V_df["V1"])
    V_df["Vunchosen"] = np.where(V_df["action"] == 1.0, V_df["V0"],V_df["V1"])
    V_df["Vchosen"] = np.where(V_df["action"] == 0.5, 0.0, V_df["Vchosen"])
    V_df["Vunchosen"] = np.where(V_df["action"] == 0.5, 0.0, V_df["Vunchosen"])
    V_df["Vchosen"] = V_df["Vchosen"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    V_df["Vunchosen"] = V_df["Vunchosen"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    functional_runs = []

    for run in range(1, 7):
        onsets = []
        durations = []
        conditions = []

        events_file = pd.read_csv(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-learn_run-{run}_events.tsv"),
            delimiter="\t",
        )
        events_file_sorted = events_file.sort_values(by=["onset"])
        events_file_sorted["trial_nr"] = events_file_sorted["trial_nr"].ffill()
        events_file_sorted["runType"] = events_file_sorted["runType"].ffill()
        run_type = events_file_sorted["runType"][0]

        confounds = pd.read_csv(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-learn_run-{run}_desc-confounds_timeseries.tsv"),
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

        physio_path = op.join(data_folder, f"ds-mlearn/derivatives/fmriprep/sub-{subject}/beh/physio/RegPhysio_sub-{subject}_run_{run}.mat")
        fn6 = glob(physio_path)
        assert len(fn6) == 1
        fn6 = fn6[0]

        physio = io.loadmat(fn6, simplify_cells=True)["physio"]["model"]
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
        run_Vchosen = V_df["Vchosen"].xs(run)
        run_Vunchosen = V_df["Vunchosen"].xs(run)
    
        pmod = [
            Bunch(name=["surprise", "Vchosen", "Vunchosen"], param=[run_surprise.values.tolist(), run_Vchosen.values.tolist(), run_Vunchosen.values.tolist()], poly=[1, 1, 1]),
            Bunch(name=["rpe"], param=[run_rpe.values.tolist()], poly=[1]),
        ]

        print(pmod)

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

        functional_run = glob(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/s6.sub-{subject}_task-learn_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii"
        ))[0]
        functional_runs.append(functional_run)

    print(subject_info)

    return subject_info, functional_runs #, V_df, rpe, surprise

def get_subject_info_model5(subject, data_folder, bestfitting_path='/mnt/d/multlearn-sns/Modelling/Fitting/bestFittingVals/'):
    from glob import glob
    import numpy as np
    import pandas as pd
    from scipy import io, stats
    from nipype.interfaces.base import Bunch
    import os.path as op

    subject_info = []
    rpe_path = (
        op.join(bestfitting_path,f"sub-{subject}/rpeBest.mat")
    )
    fn = glob(rpe_path)
    print(fn)

    assert len(fn) == 1
    fn = fn[0]

    rpe_data = np.nan_to_num(
        io.loadmat(fn, simplify_cells=True)["rpe"]["rpe"]
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

    rpe["rpe"] = rpe["rpe"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    rpe["rpe"] = np.nan_to_num(rpe["rpe"])

    surprise_path = (
        op.join(bestfitting_path,f"sub-{subject}/spe*.npy")
    )
    fn2 = glob(surprise_path)
    print(fn2)
    assert len(fn2) == 1
    fn2 = fn2[0]
    surprise_data = np.load(fn2)
    
    surprise = (
        pd.DataFrame(
            surprise_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("spe")
    )
    surprise["spe"] = surprise["spe"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    surprise["spe"] = np.nan_to_num(surprise["spe"])

    V0_path = (
        op.join(bestfitting_path,f"sub-{subject}/V0*.npy")
    )
    fn3 = glob(V0_path)
    print(fn3)
    assert len(fn3) == 1
    fn3 = fn3[0]
    V0_data = np.load(fn3)
    V0 = (
        pd.DataFrame(
            V0_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("V0")
    )

    V1_path = (
        op.join(bestfitting_path,f"sub-{subject}/V1*.npy")
    )

    fn4 = glob(V1_path)
    print(fn4)
    assert len(fn4) == 1
    fn4 = fn4[0]
    V1_data = np.load(fn4)
    V1 = (
        pd.DataFrame(
            V1_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("V1")
    )

    fn5 = op.join(
            data_folder,
            "sourcedata",
            "behavior",
            f"modified_files",
            f"modified_participant{subject}_savedValues.csv",
            )
    stimulus_info = (
    pd.read_csv(
        fn5,
        usecols=["trialNumber", "runNumber", "reward", "action", "responseTime"],
    )
    .rename(columns={"trialNumber": "trial_nr", "runNumber": "run"})
    .set_index(["run", "trial_nr"])
    .sort_index()
    )

    V_df = pd.concat([V0, V1, stimulus_info], axis=1)
    V_df["V0"] = V_df["V0"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    V_df["V1"] = V_df["V1"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    functional_runs = []

    for run in range(1, 7):
        onsets = []
        durations = []
        conditions = []

        events_file = pd.read_csv(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-learn_run-{run}_events.tsv"),
            delimiter="\t",
        )
        events_file_sorted = events_file.sort_values(by=["onset"])
        events_file_sorted["trial_nr"] = events_file_sorted["trial_nr"].ffill()
        events_file_sorted["runType"] = events_file_sorted["runType"].ffill()
        events_file_sorted.loc[events_file_sorted.trial_type == "choice", "duration"] = 0
        run_type = events_file_sorted["runType"][0]

        confounds = pd.read_csv(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-learn_run-{run}_desc-confounds_timeseries.tsv"),
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

        physio_path = op.join(data_folder, f"ds-mlearn/derivatives/fmriprep/sub-{subject}/beh/physio/RegPhysio_sub-{subject}_run_{run}.mat")
        fn6 = glob(physio_path)
        assert len(fn6) == 1
        fn6 = fn6[0]

        physio = io.loadmat(fn6, simplify_cells=True)["physio"]["model"]
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
        run_V0 = V_df["V0"].xs(run)
        run_V1 = V_df["V1"].xs(run)
    
        pmod = [
            Bunch(name=["surprise", "V0", "V1"], param=[run_surprise.values.tolist(), run_V0.values.tolist(), run_V1.values.tolist()], poly=[1, 1, 1]),
            Bunch(name=["rpe"], param=[run_rpe.values.tolist()], poly=[1]),
        ]

        print(pmod)

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

        functional_run = glob(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/s6.sub-{subject}_task-learn_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii"
        ))[0]
        functional_runs.append(functional_run)

    print(subject_info)

    return subject_info, functional_runs #, V_df, rpe, surprise


def get_subject_info_PPI(subject, data_folder, roi, variable, bestfitting_path='/mnt/d/multlearn-sns/Modelling/Fitting/bestFittingVals/'):
    from glob import glob
    import numpy as np
    import pandas as pd
    from scipy import io, stats
    from nipype.interfaces.base import Bunch
    import os.path as op

    subject_info = []
    rpe_path = (
        op.join(bestfitting_path,f"sub-{subject}/rpeBest.mat")
    )
    fn = glob(rpe_path)
    print(fn)

    assert len(fn) == 1
    fn = fn[0]

    rpe_data = np.nan_to_num(
        io.loadmat(fn, simplify_cells=True)["rpe"]["rpe"]
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

    rpe["rpe"] = rpe["rpe"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    rpe["rpe"] = np.nan_to_num(rpe["rpe"])

    urpe = rpe.copy()['rpe'].abs().to_frame('urpe')
    urpe["urpe"] = urpe["urpe"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    urpe["urpe"] = np.nan_to_num(urpe["urpe"])

    surprise_path = (
        op.join(bestfitting_path,f"sub-{subject}/spe*.npy")
    )
    fn2 = glob(surprise_path)
    print(fn2)
    assert len(fn2) == 1
    fn2 = fn2[0]
    surprise_data = np.load(fn2)
    
    surprise = (
        pd.DataFrame(
            surprise_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("spe")
    )
    surprise["spe"] = surprise["spe"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    surprise["spe"] = np.nan_to_num(surprise["spe"])

    V0_path = (
        op.join(bestfitting_path,f"sub-{subject}/V0*.npy")
    )
    fn3 = glob(V0_path)
    print(fn3)
    assert len(fn3) == 1
    fn3 = fn3[0]
    V0_data = np.load(fn3)
    V0 = (
        pd.DataFrame(
            V0_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("V0")
    )

    V1_path = (
        op.join(bestfitting_path,f"sub-{subject}/V1*.npy")
    )

    fn4 = glob(V1_path)
    print(fn4)
    assert len(fn4) == 1
    fn4 = fn4[0]
    V1_data = np.load(fn4)
    V1 = (
        pd.DataFrame(
            V1_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("V1")
    )

    fn5 = op.join(
            data_folder,
            "sourcedata",
            "behavior",
            f"modified_files",
            f"modified_participant{subject}_savedValues.csv",
            )
    stimulus_info = (
    pd.read_csv(
        fn5,
        usecols=["trialNumber", "runNumber", "reward", "action", "responseTime"],
    )
    .rename(columns={"trialNumber": "trial_nr", "runNumber": "run"})
    .set_index(["run", "trial_nr"])
    .sort_index()
    )


    V_df = pd.concat([V0, V1, stimulus_info], axis=1)
    V_df["action"] = V_df["action"].fillna(0.5)
    V_df["V"] = np.where(V_df["action"] == 0.0, V_df["V0"],V_df["V1"])
    V_df["V"] = V_df["V"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    V_df["V"] = np.where(V_df["action"] == 0.5, 0.0, V_df["V"])
    functional_runs = []

    for run in range(1, 7):
        onsets = []
        durations = []
        conditions = []

        events_file = pd.read_csv(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-learn_run-{run}_events.tsv"),
            delimiter="\t",
        )
        events_file_sorted = events_file.sort_values(by=["onset"])
        events_file_sorted["trial_nr"] = events_file_sorted["trial_nr"].ffill()
        events_file_sorted["runType"] = events_file_sorted["runType"].ffill()
        run_type = events_file_sorted["runType"][0]

        confounds = pd.read_csv(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-learn_run-{run}_desc-confounds_timeseries.tsv"),
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

        physio_path = op.join(data_folder, f"ds-mlearn/derivatives/fmriprep/sub-{subject}/beh/physio/RegPhysio_sub-{subject}_run_{run}.mat")
        fn6 = glob(physio_path)
        assert len(fn6) == 1
        fn6 = fn6[0]

        physio = io.loadmat(fn6, simplify_cells=True)["physio"]["model"]
        physio = pd.DataFrame(
            data=physio["R"],
            columns=physio["R_column_names"],
        )

        ppi_path = op.join(data_folder, "nipype/model7/1stLevel/sub-"+subject, f'PPI_{roi}_{variable}_run_{run}.mat')
        assert(op.exists(ppi_path)), f'{ppi_path} does not exist!'
        ppi = io.loadmat(ppi_path, simplify_cells=True)["PPI"]

        ppi_voi = pd.DataFrame(
            data=ppi["Y"],
            columns=["VOI"+run_type.capitalize()],
        )
        ppi_int = pd.DataFrame(
            data=ppi["ppi"],
            columns=["Interaction"+run_type.capitalize()],
        )
        
        regressors = pd.concat([ppi_voi, ppi_int, confounds, physio], axis=1)
        regressor_names = regressors.columns.values.tolist()

        for group in events_file_sorted.groupby("trial_type"):
            conditions.append(str(group[0].capitalize() + run_type.capitalize()))
            onsets.append(group[1]["onset"].tolist())
            durations.append(group[1]["duration"].tolist())
        run_urpe = urpe.xs(run)
        run_rpe = rpe.xs(run)
        run_surprise = surprise.xs(run)
        run_V = V_df["V"].xs(run)
    
        pmod = [
            Bunch(name=["surprise", "V"], param=[run_surprise.values.tolist(), run_V.values.tolist()], poly=[1, 1]),
            Bunch(name=["urpe" "rpe"], param=[run_urpe.values.tolist(), run_rpe.values.tolist()], poly=[1, 1]),
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

        functional_run = glob(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/s6.sub-{subject}_task-learn_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii"
        ))[0]
        functional_runs.append(functional_run)

    return subject_info, functional_runs #, V_df, rpe, surprise

def get_contrasts_model1(subject_info):
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

    con05 = ["surprise", "T", condition_names[4:6], [1 / 6.0, 1 / 6.0]]
    con06 = ["surprise_audio", "T", [condition_names[4]], [1 / 3.0]]
    con07 = ["surprise_tactile", "T", [condition_names[5]], [1 / 3.0]]
    con08 = [
        "surprise_audio < surprise_tactile",
        "T",
        [condition_names[4], condition_names[5]],
        [-1 / 3.0, 1 / 3.0],
    ]
    
    con09 = [
        "surprise < rpe",
        "T",
        condition_names[4:],
        [-1 / 6.0, -1 / 6.0, 1 / 6.0, 1 / 6.0],
    ]


    con10 = ["feedback", "T", condition_names[2:4], [1 / 6.0, 1 / 6.0]]
    con11 = ["choice", "T", condition_names[0:2], [1 / 6.0, 1 / 6.0]]
    con12 = [
        "feedback < choice",
        "T",
        condition_names[0:4],
        [1 / 6.0, 1 / 6.0, -1 / 6.0, -1 / 6.0],
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
    ]

    return con_list


def get_contrasts_model2(subject_info):
    from nipype.interfaces.spm import EstimateContrast
    import os

    condition_names = [
        "ChoiceAudio",
        "ChoiceTactile",
        "FeedbackAudio",
        "FeedbackTactile",
        "ChoiceAudioxsurprise^1",
        "ChoiceAudioxV^1",
        "ChoiceTactilexsurprise^1",
        "ChoiceTactilexV^1",
        "FeedbackAudioxrpe^1",
        "FeedbackTactilexrpe^1",
    ]

    con01 = ["rpe", "T", condition_names[8:10], [1 / 6.0, 1 / 6.0]]
    con02 = ["rpe_audio", "T", [condition_names[8]], [1 / 3.0]]
    con03 = ["rpe_tactile", "T", [condition_names[9]], [1 / 3.0]]
    con04 = [
        "rpe_audio < rpe_tactile",
        "T",
        [condition_names[8], condition_names[9]],
        [-1 / 3.0, 1 / 3.0],
    ]

    con05 = ["surprise", "T", [condition_names[4],condition_names[6]], [1 / 6.0, 1 / 6.0]]
    con06 = ["surprise_audio", "T", [condition_names[4]], [1 / 3.0]]
    con07 = ["surprise_tactile", "T", [condition_names[6]], [1 / 3.0]]
    con08 = [
        "surprise_audio < surprise_tactile",
        "T",
        [condition_names[4], condition_names[6]],
        [-1 / 3.0, 1 / 3.0],
    ]

    con09 = ["value", "T", [condition_names[5],condition_names[7]], [1 / 6.0, 1 / 6.0]]
    con10 = ["value_audio", "T", [condition_names[5]], [1 / 3.0]]
    con11 = ["value_tactile", "T", [condition_names[7]], [1 / 3.0]]
    con12 = [
        "value_audio < value_tactile",
        "T",
        [condition_names[5], condition_names[7]],
        [-1 / 3.0, 1 / 3.0],
    ]
    
    con13 = [
        "surprise < rpe",
        "T",
        [condition_names[4], condition_names[6],condition_names[8],condition_names[9]],
        [-1 / 6.0, -1 / 6.0, 1 / 6.0, 1 / 6.0],
    ]

    con14 = ["feedback", "T", condition_names[2:4], [1 / 6.0, 1 / 6.0]]
    con15 = ["choice", "T", condition_names[0:2], [1 / 6.0, 1 / 6.0]]
    con16 = [
        "feedback < choice",
        "T",
        condition_names[0:4],
        [1 / 6.0, 1 / 6.0, -1 / 6.0, -1 / 6.0],
    ]
    con17 = [
        "choice_audio > choice_tactile",
        "T",
        condition_names[0:2],
        [1 / 3.0, -1 / 3.0],
    ]
    con18 = [
        "feedback_audio < feedback_visual",
        "T",
        condition_names[2:4],
        [1 / 3.0, -1 / 3.0, ],
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
        con18
    ]

    return con_list

def get_contrasts_model3(subject_info):
    from nipype.interfaces.spm import EstimateContrast
    import os

    condition_names = [
        "ChoiceAudio",
        "ChoiceTactile",
        "FeedbackAudio",
        "FeedbackTactile",
        "ChoiceAudioxsurprise^1",
        "ChoiceAudioxV^1",
        "ChoiceTactilexsurprise^1",
        "ChoiceTactilexV^1",
        "FeedbackAudioxrpe^1",
        "FeedbackTactilexrpe^1",
    ]

    con01 = ["rpe", "T", condition_names[8:10], [1 / 6.0, 1 / 6.0]]
    con02 = ["rpe_audio", "T", [condition_names[8]], [1 / 3.0]]
    con03 = ["rpe_tactile", "T", [condition_names[9]], [1 / 3.0]]
    con04 = [
        "rpe_audio < rpe_tactile",
        "T",
        [condition_names[8], condition_names[9]],
        [-1 / 3.0, 1 / 3.0],
    ]

    con05 = ["surprise", "T", [condition_names[4],condition_names[6]], [1 / 6.0, 1 / 6.0]]
    con06 = ["surprise_audio", "T", [condition_names[4]], [1 / 3.0]]
    con07 = ["surprise_tactile", "T", [condition_names[6]], [1 / 3.0]]
    con08 = [
        "surprise_audio < surprise_tactile",
        "T",
        [condition_names[4], condition_names[6]],
        [-1 / 3.0, 1 / 3.0],
    ]

    con09 = ["value", "T", [condition_names[5],condition_names[7]], [1 / 6.0, 1 / 6.0]]
    con10 = ["value_audio", "T", [condition_names[5]], [1 / 3.0]]
    con11 = ["value_tactile", "T", [condition_names[7]], [1 / 3.0]]
    con12 = [
        "value_audio < value_tactile",
        "T",
        [condition_names[5], condition_names[7]],
        [-1 / 3.0, 1 / 3.0],
    ]
    
    con13 = [
        "surprise < rpe",
        "T",
        [condition_names[4], condition_names[6],condition_names[8],condition_names[9]],
        [-1 / 6.0, -1 / 6.0, 1 / 6.0, 1 / 6.0],
    ]

    con14 = ["feedback", "T", condition_names[2:4], [1 / 6.0, 1 / 6.0]]
    con15 = ["choice", "T", condition_names[0:2], [1 / 6.0, 1 / 6.0]]
    con16 = [
        "feedback < choice",
        "T",
        condition_names[0:4],
        [1 / 6.0, 1 / 6.0, -1 / 6.0, -1 / 6.0],
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
    ]

    return con_list

def get_contrasts_model4(subject_info):
    from nipype.interfaces.spm import EstimateContrast
    import os

    condition_names = [
        "ChoiceAudio",
        "ChoiceTactile",
        "FeedbackAudio",
        "FeedbackTactile",
        "ChoiceAudioxsurprise^1",
        "ChoiceAudioxVchosen^1",
        "ChoiceAudioxVunchosen^1",
        "ChoiceTactilexsurprise^1",
        "ChoiceTactilexVchosen^1",
        "ChoiceTactilexVunchosen^1",
        "FeedbackAudioxrpe^1",
        "FeedbackTactilexrpe^1",
    ]

    con01 = ["rpe", "T", condition_names[9:11], [1 / 6.0, 1 / 6.0]]
    con02 = ["rpe_audio", "T", [condition_names[10]], [1 / 3.0]]
    con03 = ["rpe_tactile", "T", [condition_names[11]], [1 / 3.0]]
    con04 = [
        "rpe_audio < rpe_tactile",
        "T",
        [condition_names[10], condition_names[11]],
        [-1 / 3.0, 1 / 3.0],
    ]

    con05 = ["surprise", "T", [condition_names[4],condition_names[7]], [1 / 6.0, 1 / 6.0]]
    con06 = ["surprise_audio", "T", [condition_names[4]], [1 / 3.0]]
    con07 = ["surprise_tactile", "T", [condition_names[7]], [1 / 3.0]]
    con08 = [
        "surprise_audio < surprise_tactile",
        "T",
        [condition_names[4], condition_names[7]],
        [-1 / 3.0, 1 / 3.0],
    ]

    con09 = ["Vchosen", "T", [condition_names[5],condition_names[8]], [1 / 6.0, 1 / 6.0]]
    con10 = ["Vchosen_audio", "T", [condition_names[5]], [1 / 3.0]]
    con11 = ["Vchosen_tactile", "T", [condition_names[8]], [1 / 3.0]]
    con12 = [
        "Vchosen_audio < Vchosen_tactile",
        "T",
        [condition_names[5], condition_names[8]],
        [-1 / 3.0, 1 / 3.0],
    ]

    con13 = ["Vunchosen", "T", [condition_names[6],condition_names[9]], [1 / 6.0, 1 / 6.0]]
    con14 = ["Vunchosen_audio", "T", [condition_names[6]], [1 / 3.0]]
    con15 = ["Vunchosen_tactile", "T", [condition_names[9]], [1 / 3.0]]
    con16 = [
        "Vunchosen_audio < Vunchosen_tactile",
        "T",
        [condition_names[6], condition_names[9]],
        [-1 / 3.0, 1 / 3.0],
    ]

    
    
    con17 = [
        "surprise < rpe",
        "T",
        [condition_names[4], condition_names[7],condition_names[10],condition_names[11]],
        [-1 / 6.0, -1 / 6.0, 1 / 6.0, 1 / 6.0],
    ]
    
    

    con18 = ["feedback", "T", condition_names[2:4], [1 / 6.0, 1 / 6.0]]
    con19 = ["choice", "T", condition_names[0:2], [1 / 6.0, 1 / 6.0]]
    con20 = [
        "feedback < choice",
        "T",
        condition_names[0:4],
        [1 / 6.0, 1 / 6.0, -1 / 6.0, -1 / 6.0],
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
        con20
    ]

    return con_list

def get_contrasts_model5(subject_info):
    from nipype.interfaces.spm import EstimateContrast
    import os

    condition_names = [
        "ChoiceAudio",
        "ChoiceTactile",
        "FeedbackAudio",
        "FeedbackTactile",
        "ChoiceAudioxsurprise^1",
        "ChoiceAudioxV0^1",
        "ChoiceAudioxV1^1",
        "ChoiceTactilexsurprise^1",
        "ChoiceTactilexV0^1",
        "ChoiceTactilexV1^1",
        "FeedbackAudioxrpe^1",
        "FeedbackTactilexrpe^1",
    ]

    con01 = ["rpe", "T", condition_names[9:11], [1 / 6.0, 1 / 6.0]]
    con02 = ["rpe_audio", "T", [condition_names[10]], [1 / 3.0]]
    con03 = ["rpe_tactile", "T", [condition_names[11]], [1 / 3.0]]
    con04 = [
        "rpe_audio < rpe_tactile",
        "T",
        [condition_names[10], condition_names[11]],
        [-1 / 3.0, 1 / 3.0],
    ]

    con05 = ["surprise", "T", [condition_names[4],condition_names[7]], [1 / 6.0, 1 / 6.0]]
    con06 = ["surprise_audio", "T", [condition_names[4]], [1 / 3.0]]
    con07 = ["surprise_tactile", "T", [condition_names[7]], [1 / 3.0]]
    con08 = [
        "surprise_audio < surprise_tactile",
        "T",
        [condition_names[4], condition_names[7]],
        [-1 / 3.0, 1 / 3.0],
    ]

    con09 = ["V0", "T", [condition_names[5],condition_names[8]], [1 / 6.0, 1 / 6.0]]
    con10 = ["V0_audio", "T", [condition_names[5]], [1 / 3.0]]
    con11 = ["V0_tactile", "T", [condition_names[8]], [1 / 3.0]]
    con12 = [
        "V0_audio < V0_tactile",
        "T",
        [condition_names[5], condition_names[8]],
        [-1 / 3.0, 1 / 3.0],
    ]

    con13 = ["V1", "T", [condition_names[6],condition_names[9]], [1 / 6.0, 1 / 6.0]]
    con14 = ["V1_audio", "T", [condition_names[6]], [1 / 3.0]]
    con15 = ["V1_tactile", "T", [condition_names[9]], [1 / 3.0]]
    con16 = [
        "V1_audio < V1_tactile",
        "T",
        [condition_names[6], condition_names[9]],
        [-1 / 3.0, 1 / 3.0],
    ]

    
    
    con17 = [
        "surprise < rpe",
        "T",
        [condition_names[4], condition_names[7],condition_names[10],condition_names[11]],
        [-1 / 6.0, -1 / 6.0, 1 / 6.0, 1 / 6.0],
    ]
    
    

    con18 = ["feedback", "T", condition_names[2:4], [1 / 6.0, 1 / 6.0]]
    con19 = ["choice", "T", condition_names[0:2], [1 / 6.0, 1 / 6.0]]
    con20 = [
        "feedback < choice",
        "T",
        condition_names[0:4],
        [1 / 6.0, 1 / 6.0, -1 / 6.0, -1 / 6.0],
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
        con20
    ]

    return con_list

def get_subject_info_model6(subject, data_folder, bestfitting_path='/mnt/d/multlearn-sns/Modelling/Fitting/bestFittingVals/'):
    from glob import glob
    import numpy as np
    import pandas as pd
    from scipy import io, stats
    from nipype.interfaces.base import Bunch
    import os.path as op

    subject_info = []
    rpe_path = (
        op.join(bestfitting_path,f"sub-{subject}/rpeBest.mat")
    )
    fn = glob(rpe_path)
    print(fn)

    assert len(fn) == 1
    fn = fn[0]

    rpe_data = np.nan_to_num(
        io.loadmat(fn, simplify_cells=True)["rpe"]["rpe"]
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

    urpe = rpe.copy()['rpe'].abs().to_frame('urpe')
    # rpe["rpe"] = rpe["rpe"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    # rpe["rpe"] = np.nan_to_num(rpe["rpe"])
    urpe['urpe'] = urpe['urpe'].groupby('run').transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    urpe['urpe'] = np.nan_to_num(urpe['urpe'])

    surprise_path = (
        op.join(bestfitting_path,f"sub-{subject}/spe*.npy")
    )
    fn2 = glob(surprise_path)
    print(fn2)
    assert len(fn2) == 1
    fn2 = fn2[0]
    surprise_data = np.load(fn2)
    
    surprise = (
        pd.DataFrame(
            surprise_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("spe")
    )
    surprise["spe"] = surprise["spe"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    surprise["spe"] = np.nan_to_num(surprise["spe"])

    V0_path = (
        op.join(bestfitting_path,f"sub-{subject}/V0*.npy")
    )
    fn3 = glob(V0_path)
    print(fn3)
    assert len(fn3) == 1
    fn3 = fn3[0]
    V0_data = np.load(fn3)
    V0 = (
        pd.DataFrame(
            V0_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("V0")
    )

    V1_path = (
        op.join(bestfitting_path,f"sub-{subject}/V1*.npy")
    )

    fn4 = glob(V1_path)
    print(fn4)
    assert len(fn4) == 1
    fn4 = fn4[0]
    V1_data = np.load(fn4)
    V1 = (
        pd.DataFrame(
            V1_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("V1")
    )

    fn5 = op.join(
            data_folder,
            "sourcedata",
            "behavior",
            f"modified_files",
            f"modified_participant{subject}_savedValues.csv",
            )
    stimulus_info = (
    pd.read_csv(
        fn5,
        usecols=["trialNumber", "runNumber", "reward", "action", "responseTime"],
    )
    .rename(columns={"trialNumber": "trial_nr", "runNumber": "run"})
    .set_index(["run", "trial_nr"])
    .sort_index()
    )

    V_df = pd.concat([V0, V1, stimulus_info], axis=1)
    V_df["action"] = V_df["action"].fillna(0.5)
    V_df["V"] = np.where(V_df["action"] == 0.0, V_df["V0"],V_df["V1"])
    V_df["V"] = V_df["V"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    V_df["V"] = np.where(V_df["action"] == 0.5, 0.0, V_df["V"])
    functional_runs = []

    for run in range(1, 7):
        onsets = []
        durations = []
        conditions = []

        events_file = pd.read_csv(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-learn_run-{run}_events.tsv"),
            delimiter="\t",
        )
        events_file_sorted = events_file.sort_values(by=["onset"])
        events_file_sorted["trial_nr"] = events_file_sorted["trial_nr"].ffill()
        events_file_sorted["runType"] = events_file_sorted["runType"].ffill()
        run_type = events_file_sorted["runType"][0]

        confounds = pd.read_csv(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-learn_run-{run}_desc-confounds_timeseries.tsv"),
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

        physio_path = op.join(data_folder, f"ds-mlearn/derivatives/fmriprep/sub-{subject}/beh/physio/RegPhysio_sub-{subject}_run_{run}.mat")
        fn6 = glob(physio_path)
        assert len(fn6) == 1
        fn6 = fn6[0]

        physio = io.loadmat(fn6, simplify_cells=True)["physio"]["model"]
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
        run_urpe = urpe.xs(run)
        run_surprise = surprise.xs(run)
        run_V = V_df["V"].xs(run)
    
        pmod = [
            Bunch(name=["surprise", "V"], param=[run_surprise.values.tolist(), run_V.values.tolist()], poly=[1, 1]),
            Bunch(name=["urpe"], param=[run_urpe.values.tolist()], poly=[1]),
        ]

        print(pmod)

        subject_info.insert(
            run - 1,
            Bunch(
                conditions=conditions,
                onsets=onsets,
                durations=durations,
                pmod=pmod,
                tmod=None,
                regressors=regressors.values.T.tolist(),
                regressor_names=regressor_names,
            ),
        )

        functional_run = glob(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/s6.sub-{subject}_task-learn_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii"
        ))[0]
        functional_runs.append(functional_run)

    print(subject_info)

    return subject_info, functional_runs #, V_df, rpe, surprise

def get_contrasts_ROI(subject_info):
    from nipype.interfaces.spm import EstimateContrast
    import os

    condition_names = [
        "ChoiceAudio",
        "ChoiceTactile",
        "FeedbackAudio",
        "FeedbackTactile",
        "ChoiceAudioxsurprise^1",
        "ChoiceAudioxV^1",
        "ChoiceTactilexsurprise^1",
        "ChoiceTactilexV^1",
        "FeedbackAudioxrpe^1",
        "FeedbackTactilexrpe^1",
    ]

    con01 = [
        "rpe_audio < rpe_tactile",
        "T",
        [condition_names[8], condition_names[9]],
        [-1 / 3.0, 1 / 3.0],
    ]

    con02 = [
        "surprise_audio < surprise_tactile",
        "T",
        [condition_names[4], condition_names[6]],
        [-1 / 3.0, 1 / 3.0],
    ]

    con03 = [
        "value_audio < value_tactile",
        "T",
        [condition_names[5], condition_names[7]],
        [-1 / 3.0, 1 / 3.0],
    ]
    
    
    con_list = [
        con01,
        con02,
        con03,
    ]

    return con_list

def get_contrasts_PPI(subject_info):
    from nipype.interfaces.spm import EstimateContrast
    import os

    condition_names = [
        "InteractionAudio",
        "InteractionTactile",
        "VOIAudio",
        "VOITactile"
    ]

    con01 = [
        "Interaction",
        "T",
        [condition_names[0], condition_names[1]],
        [1 / 3.0, 1 / 3.0],
    ]

    con02 = [
        "Interaction_audio < Interaction_tactile",
        "T",
        [condition_names[0], condition_names[1]],
        [-1 / 3.0, 1 / 3.0],
    ]

    con03 = [
        "Interaction_audio",
        "T",
        [condition_names[0]],
        [1 / 3.0],
    ]

    con04 = [
        "Interaction_tactile",
        "T",
        [condition_names[1]],
        [1 / 3.0],
    ]

    con05 = [
        "Connectivity Audio",
        "T",
        [condition_names[2]],
        [1 / 3.0],
    ]

    con06 = [
        "Connectivity Tactile",
        "T",
        [condition_names[3]],
        [1 / 3.0],
    ]

    con07 = [
        "Connectivity Audio < Connectivity Tactile",
        "T",
        [condition_names[2], condition_names[3]],
        [-1./3., 1 / 3.0],
    ]
    
    
    con_list = [
        con01,
        con02,
        con03,
        con04,
        con05,
        con06,
        con07
    ]

    return con_list

def get_contrasts_model6(subject_info):
    from nipype.interfaces.spm import EstimateContrast
    import os

    condition_names = [
        "ChoiceAudio",
        "ChoiceTactile",
        "FeedbackAudio",
        "FeedbackTactile",
        "ChoiceAudioxsurprise^1",
        "ChoiceAudioxV^1",
        "ChoiceTactilexsurprise^1",
        "ChoiceTactilexV^1",
        "FeedbackAudioxurpe^1",
        "FeedbackTactilexurpe^1",
    ]

    con01 = ["urpe", "T", condition_names[8:10], [1 / 6.0, 1 / 6.0]]
    con02 = ["urpe_audio", "T", [condition_names[8]], [1 / 3.0]]
    con03 = ["urpe_tactile", "T", [condition_names[9]], [1 / 3.0]]
    con04 = [
        "urpe_audio < urpe_tactile",
        "T",
        [condition_names[8], condition_names[9]],
        [-1 / 3.0, 1 / 3.0],
    ]

    con05 = ["surprise", "T", [condition_names[4],condition_names[6]], [1 / 6.0, 1 / 6.0]]
    con06 = ["surprise_audio", "T", [condition_names[4]], [1 / 3.0]]
    con07 = ["surprise_tactile", "T", [condition_names[6]], [1 / 3.0]]
    con08 = [
        "surprise_audio < surprise_tactile",
        "T",
        [condition_names[4], condition_names[6]],
        [-1 / 3.0, 1 / 3.0],
    ]

    con09 = ["value", "T", [condition_names[5],condition_names[7]], [1 / 6.0, 1 / 6.0]]
    con10 = ["value_audio", "T", [condition_names[5]], [1 / 3.0]]
    con11 = ["value_tactile", "T", [condition_names[7]], [1 / 3.0]]
    con12 = [
        "value_audio < value_tactile",
        "T",
        [condition_names[5], condition_names[7]],
        [-1 / 3.0, 1 / 3.0],
    ]
    
    con13 = [
        "surprise < urpe",
        "T",
        [condition_names[4], condition_names[6],condition_names[8],condition_names[9]],
        [-1 / 6.0, -1 / 6.0, 1 / 6.0, 1 / 6.0],
    ]

    con14 = ["feedback", "T", condition_names[2:4], [1 / 6.0, 1 / 6.0]]
    con15 = ["choice", "T", condition_names[0:2], [1 / 6.0, 1 / 6.0]]
    con16 = [
        "feedback < choice",
        "T",
        condition_names[0:4],
        [1 / 6.0, 1 / 6.0, -1 / 6.0, -1 / 6.0],
    ]
    con17 = [
        "choice_audio > choice_tactile",
        "T",
        condition_names[0:2],
        [1 / 3.0, -1 / 3.0],
    ]
    con18 = [
        "feedback_audio < feedback_visual",
        "T",
        condition_names[2:4],
        [1 / 3.0, -1 / 3.0, ],
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
        con18
    ]

    return con_list



def get_subject_info_model7(subject, data_folder, bestfitting_path='/mnt/d/multlearn-sns/Modelling/Fitting/bestFittingVals/'):
    from glob import glob
    import numpy as np
    import pandas as pd
    from scipy import io, stats
    from nipype.interfaces.base import Bunch
    import os.path as op

    subject_info = []
    rpe_path = (
        op.join(bestfitting_path,f"sub-{subject}/rpeBest.mat")
    )
    fn = glob(rpe_path)
    print(fn)

    assert len(fn) == 1
    fn = fn[0]

    rpe_data = np.nan_to_num(
        io.loadmat(fn, simplify_cells=True)["rpe"]["rpe"]
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

    urpe = rpe.copy()['rpe'].abs().to_frame('urpe')
    rpe["rpe"] = rpe["rpe"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    rpe["rpe"] = np.nan_to_num(rpe["rpe"])
    urpe['urpe'] = urpe['urpe'].groupby('run').transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    urpe['urpe'] = np.nan_to_num(urpe['urpe'])

    surprise_path = (
        op.join(bestfitting_path,f"sub-{subject}/spe*.npy")
    )
    fn2 = glob(surprise_path)
    print(fn2)
    assert len(fn2) == 1
    fn2 = fn2[0]
    surprise_data = np.load(fn2)
    
    surprise = (
        pd.DataFrame(
            surprise_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("spe")
    )
    surprise["spe"] = surprise["spe"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    surprise["spe"] = np.nan_to_num(surprise["spe"])

    V0_path = (
        op.join(bestfitting_path,f"sub-{subject}/V0*.npy")
    )
    fn3 = glob(V0_path)
    print(fn3)
    assert len(fn3) == 1
    fn3 = fn3[0]
    V0_data = np.load(fn3)
    V0 = (
        pd.DataFrame(
            V0_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("V0")
    )

    V1_path = (
        op.join(bestfitting_path,f"sub-{subject}/V1*.npy")
    )

    fn4 = glob(V1_path)
    print(fn4)
    assert len(fn4) == 1
    fn4 = fn4[0]
    V1_data = np.load(fn4)
    V1 = (
        pd.DataFrame(
            V1_data,
            index=pd.Index(np.arange(1, 6 + 1), name="run"),
            columns=pd.Index(np.arange(1, 60 + 1), name="trial_nr"),
        )
        .stack()
        .to_frame("V1")
    )

    fn5 = op.join(
            data_folder,
            "sourcedata",
            "behavior",
            f"modified_files",
            f"modified_participant{subject}_savedValues.csv",
            )
    stimulus_info = (
    pd.read_csv(
        fn5,
        usecols=["trialNumber", "runNumber", "reward", "action", "responseTime"],
    )
    .rename(columns={"trialNumber": "trial_nr", "runNumber": "run"})
    .set_index(["run", "trial_nr"])
    .sort_index()
    )

    V_df = pd.concat([V0, V1, stimulus_info], axis=1)
    V_df["action"] = V_df["action"].fillna(0.5)
    V_df["V"] = np.where(V_df["action"] == 0.0, V_df["V0"],V_df["V1"])
    V_df["V"] = V_df["V"].groupby("run").transform(lambda x: stats.zscore(x - np.nanmean(x)) if x is not np.nan else x)
    V_df["V"] = np.where(V_df["action"] == 0.5, 0.0, V_df["V"])
    functional_runs = []

    for run in range(1, 7):
        onsets = []
        durations = []
        conditions = []

        events_file = pd.read_csv(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-learn_run-{run}_events.tsv"),
            delimiter="\t",
        )
        events_file_sorted = events_file.sort_values(by=["onset"])
        events_file_sorted["trial_nr"] = events_file_sorted["trial_nr"].ffill()
        events_file_sorted["runType"] = events_file_sorted["runType"].ffill()
        run_type = events_file_sorted["runType"][0]

        confounds = pd.read_csv(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-learn_run-{run}_desc-confounds_timeseries.tsv"),
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

        physio_path = op.join(data_folder, f"ds-mlearn/derivatives/fmriprep/sub-{subject}/beh/physio/RegPhysio_sub-{subject}_run_{run}.mat")
        fn6 = glob(physio_path)
        assert len(fn6) == 1
        fn6 = fn6[0]

        physio = io.loadmat(fn6, simplify_cells=True)["physio"]["model"]
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
        run_urpe = urpe.xs(run)
        run_rpe = rpe.xs(run)
        run_surprise = surprise.xs(run)
        run_V = V_df["V"].xs(run)
    
        pmod = [
            Bunch(name=["surprise", "V"], param=[run_surprise.values.tolist(), run_V.values.tolist()], poly=[1, 1]),
            Bunch(name=["urpe", "rpe"], param=[run_urpe.values.tolist(), run_rpe.values.tolist()], poly=[1, 1]),
        ]

        print(pmod)

        subject_info.insert(
            run - 1,
            Bunch(
                conditions=conditions,
                onsets=onsets,
                durations=durations,
                pmod=pmod,
                tmod=None,
                regressors=regressors.values.T.tolist(),
                regressor_names=regressor_names,
            ),
        )

        functional_run = glob(op.join(data_folder,
            f"ds-mlearn/derivatives/fmriprep/sub-{subject}/func/s6.sub-{subject}_task-learn_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii"
        ))[0]
        functional_runs.append(functional_run)

    print(subject_info)

    return subject_info, functional_runs #, V_df, rpe, surprise

def get_contrasts_model7(subject_info):
    from nipype.interfaces.spm import EstimateContrast
    import os

    condition_names = [
        "ChoiceAudio",
        "ChoiceTactile",
        "FeedbackAudio",
        "FeedbackTactile",
        "ChoiceAudioxsurprise^1",
        "ChoiceAudioxV^1",
        "ChoiceTactilexsurprise^1",
        "ChoiceTactilexV^1",
        "FeedbackAudioxurpe^1",
        "FeedbackTactilexurpe^1",
        "FeedbackAudioxrpe^1",
        "FeedbackTactilexrpe^1",
    ]

    con01 = ["urpe", "T", condition_names[8:10], [1 / 6.0, 1 / 6.0]]
    con02 = ["urpe_audio", "T", [condition_names[8]], [1 / 3.0]]
    con03 = ["urpe_tactile", "T", [condition_names[9]], [1 / 3.0]]
    con04 = [
        "urpe_audio < urpe_tactile",
        "T",
        [condition_names[8], condition_names[9]],
        [-1 / 3.0, 1 / 3.0],
    ]

    con05 = ["surprise", "T", [condition_names[4],condition_names[6]], [1 / 6.0, 1 / 6.0]]
    con06 = ["surprise_audio", "T", [condition_names[4]], [1 / 3.0]]
    con07 = ["surprise_tactile", "T", [condition_names[6]], [1 / 3.0]]
    con08 = [
        "surprise_audio < surprise_tactile",
        "T",
        [condition_names[4], condition_names[6]],
        [-1 / 3.0, 1 / 3.0],
    ]

    con09 = ["value", "T", [condition_names[5],condition_names[7]], [1 / 6.0, 1 / 6.0]]
    con10 = ["value_audio", "T", [condition_names[5]], [1 / 3.0]]
    con11 = ["value_tactile", "T", [condition_names[7]], [1 / 3.0]]
    con12 = [
        "value_audio < value_tactile",
        "T",
        [condition_names[5], condition_names[7]],
        [-1 / 3.0, 1 / 3.0],
    ]
    
    con13 = [
        "surprise < urpe",
        "T",
        [condition_names[4], condition_names[6],condition_names[8],condition_names[9]],
        [-1 / 6.0, -1 / 6.0, 1 / 6.0, 1 / 6.0],
    ]

    con14 = ["feedback", "T", condition_names[2:4], [1 / 6.0, 1 / 6.0]]
    con15 = ["choice", "T", condition_names[0:2], [1 / 6.0, 1 / 6.0]]
    con16 = [
        "feedback < choice",
        "T",
        condition_names[0:4],
        [1 / 6.0, 1 / 6.0, -1 / 6.0, -1 / 6.0],
    ]
    con17 = [
        "choice_audio > choice_tactile",
        "T",
        condition_names[0:2],
        [1 / 3.0, -1 / 3.0],
    ]
    con18 = [
        "feedback_audio < feedback_visual",
        "T",
        condition_names[2:4],
        [1 / 3.0, -1 / 3.0, ],
    ]

    con19 = ["rpe", "T", condition_names[10:12], [1 / 6.0, 1 / 6.0]]
    con20 = ["rpe_audio", "T", [condition_names[10]], [1 / 3.0]]
    con21 = ["rpe_tactile", "T", [condition_names[11]], [1 / 3.0]]
    con22 = [
        "rpe_audio < rpe_tactile",
        "T",
        [condition_names[10], condition_names[11]],
        [-1 / 3.0, 1 / 3.0],
    ]
    con23 = [
        "choice_tactile > choice_audio",
        "T",
        condition_names[0:2],
        [-1 / 3.0, 1 / 3.0],
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
        con23
    ]

    return con_list