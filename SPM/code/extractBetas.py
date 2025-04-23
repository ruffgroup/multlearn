import nibabel as nib
import numpy as np
import matplotlib.backends.backend_pdf
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
from scipy import io, stats
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, traits, File, Directory, TraitedSpec
import nipype.interfaces.matlab as matlab
from nipype.interfaces.matlab import MatlabCommand
from nipype import Node, Workflow
import shutil
import seaborn as sns
from itertools import chain
import pandas as pd

nr_rpe_mask_neg = [0, 3, 2, 4, 1]
nr_rpe_mask_pos = [0, 2, 7, 4, 3, 8, 5]
nr_surprise_mask_pos = [0, 1, 2, 3]

#roi_masks_rpe_neg = ["/shares/zne.uzh/multlearn/nipype/model2/ROI/cluster_{}_con1_4_0_neg.nii".format(str(nr)) for nr in nr_rpe_mask_neg]
#roi_masks_rpe_pos = ["/shares/zne.uzh/multlearn/nipype/model2/ROI/cluster_{}_con1_8_0_pos.nii".format(str(nr)) for nr in nr_rpe_mask_pos]
roi_masks_surprise_pos = ["/shares/zne.uzh/multlearn/nipype/model2/ROI/cluster_{}_con5_3_1_pos.nii".format(str(nr)) for nr in nr_surprise_mask_pos]
roi_masks = roi_masks_surprise_pos #roi_masks_rpe_neg+roi_masks_rpe_pos+
#roi_names_rpe_neg = ["Insula_r"]
#roi_names_rpe_pos = ["RL_Caudate_Putamen_r", "RL_VisualCortex_r", "RL_Caudate_Putamen_l",
#                      "RL_VMPFC", "RL_VisualCortex_l", "RL_Cerebellum_r", "RL_ACC_l",
#                      "RL_AngularGyrus_l", "RL_LingualGyrus_l"]
roi_names_surprise_pos = ["SL_Precuneus", "SL_AngularGyrus_TPJ_l", "SL_DLPFC_DMPFC_l", "SL_AngularGyrus_r", "SL_M1_l"]
roi_names = roi_names_surprise_pos#roi_names_rpe_neg+roi_names_rpe_pos+
model_path = "/shares/zne.uzh/multlearn/nipype/model2"
subject_list = range(1,65)
subject_list = [str(sub).zfill(2) for sub in subject_list if sub not in [8, 13, 16, 31, 32, 44]]

all_betas_df = pd.DataFrame(columns=roi_names)
for idx, roi_mask in enumerate(roi_masks):
    ROI = roi_mask.split("/")[-1].split(".nii")[0]
    cluster_nr = str(int(ROI.split("_")[1])+1)
    con = roi_mask.split("/")[-1].split("_")[-4]
    temp = re.compile("([a-zA-Z]+)([0-9]+)")
    con_nr = temp.match(con).groups()[1]

    condition_names = [
            "ChoiceAudioxsurprise^1",
            "ChoiceTactilexsurprise^1",
            "FeedbackAudioxrpe^1",
            "FeedbackTactilexrpe^1",
        ]

    # Get the image of the roi (roi_mask) and load the correct betas depending on contrast
    roi_data = get_data(nib.load(roi_mask))

    group_ROI_audio_betas = list()
    group_ROI_tactile_betas = list()
    for sub in subject_list:
        print("sub: ",sub)
        SPM_path = (
            op.join(model_path,"1stLevel",f"sub-{sub}/SPM.mat")
        )
        
        fn = glob(SPM_path)
        assert len(fn) == 1
        fn = fn[0]

        SPM_column_data = io.loadmat(fn, simplify_cells=True, squeeze_me=True)["SPM"]["xX"]["name"]
        audio_choice_cols = [idx+1 for idx,col in enumerate(SPM_column_data) if condition_names[0] in col]
        audio_feedback_cols = [idx+1 for idx,col in enumerate(SPM_column_data) if condition_names[2] in col]
        tactile_choice_cols = [idx+1 for idx,col in enumerate(SPM_column_data) if condition_names[1] in col]
        tactile_feedback_cols = [idx+1 for idx,col in enumerate(SPM_column_data) if condition_names[3] in col]

        Ind_ROI_audio_betas = list()
        Ind_ROI_tactile_betas = list()
        if int(con_nr) == 1:
            for audio_beta_img in audio_feedback_cols:
                fmri_img = op.join(model_path, "1stLevel/sub-"+sub,f"beta_{audio_beta_img:04d}.nii")
                print(fmri_img)
                fmri_data = get_data(nib.load(fmri_img))
                # Get the masked fMRI data within the ROI
                masked_data = fmri_data[roi_data > 0]
                # Average the beta values within the ROI
                Ind_ROI_audio_betas.append(np.mean(masked_data))

            for tactile_beta_img in tactile_feedback_cols:
                fmri_img = op.join(model_path, "1stLevel/sub-"+sub,f"beta_{tactile_beta_img:04d}.nii")
                fmri_data = get_data(nib.load(fmri_img))
                masked_data = fmri_data[roi_data > 0]
                Ind_ROI_tactile_betas.append(np.mean(masked_data))
            print(Ind_ROI_audio_betas)
            print(Ind_ROI_tactile_betas)

        elif int(con_nr) == 5:
            for audio_beta_img in audio_choice_cols:
                fmri_img = op.join(model_path, "1stLevel/sub-"+sub,f"beta_{audio_beta_img:04d}.nii")
                fmri_data = get_data(nib.load(fmri_img))
                # Get the masked fMRI data within the ROI
                masked_data = fmri_data[roi_data > 0]
                # Average the beta values within the ROI
                Ind_ROI_audio_betas.append(np.mean(masked_data))

            for tactile_beta_img in tactile_choice_cols:
                fmri_img = op.join(model_path, "1stLevel/sub-"+sub,f"beta_{tactile_beta_img:04d}.nii")
                fmri_data = get_data(nib.load(fmri_img))
                masked_data = fmri_data[roi_data > 0]
                Ind_ROI_tactile_betas.append(np.mean(masked_data))

        group_ROI_audio_betas.append(np.mean(Ind_ROI_audio_betas))
        group_ROI_tactile_betas.append(np.mean(Ind_ROI_tactile_betas))


    fig, ax = plt.subplots(figsize=(6, 7))
    sns.despine()
    sns.set_theme(style="ticks", rc={"lines.linewidth": 2.5})
    sns.pointplot(
    data=[group_ROI_audio_betas, group_ROI_tactile_betas],
    linestyle="none", errorbar="se", color='black',
    marker="_", join=False, ax=ax, zorder=100,
    )
    
    sns.stripplot(data=[group_ROI_audio_betas, group_ROI_tactile_betas], color="grey", zorder=1, s=20,alpha=0.8,jitter=False, marker="x", ax=ax)
    ax.axhline(y=0, linewidth=2, color='black', ls='--')
    
    if int(con_nr) == 1:
        plt.title("Reward prediction error", fontsize=20, weight="bold")
    elif int(con_nr) == 5:
        plt.title("Surprise", fontsize=20, weight="bold")
    # Increase size of tick labels
    plt.yticks(fontsize=18, weight="bold")
    plt.ylim((-1,1))

    plt.ylabel("Neural betas", fontsize=20, weight="bold")
    # Set x tick labels
    plt.xticks([0, 1], ["Audio", "Tactile"], fontsize=18, weight="bold")
    fig.tight_layout()

    save_name = "{0}_{1}_plot".format(ROI, "betas")

    file_path = os.path.join(model_path, save_name)

    pdf = matplotlib.backends.backend_pdf.PdfPages(file_path+".pdf")
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
    pdf.close()
    plt.close("all")

    

   

