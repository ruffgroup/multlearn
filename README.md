# Multisensory Learning: An fMRI Study of Audiovisual and Visuotactile Learning (2021-2025)

**Research team:** Saurabh Bedi, Ella Casimiro & Gilles de Hollander

## Overview

This repository contains the experimental code, analysis pipelines, and modeling frameworks for a comprehensive fMRI study investigating how humans learn associations between multisensory stimuli. The study examines both **reinforcement learning** (reward-driven learning) and **statistical learning** (structure discovery) across two sensory modality pairs:

1. **Audiovisual learning** – Learning associations between auditory and visual stimuli
2. **Visuotactile learning** – Learning associations between visual and tactile stimuli

Participants perform a multisensory learning task while undergoing fMRI scanning, with concurrent behavioral tracking and eye-tracking measurements. The project includes comprehensive computational modeling using reinforcement learning frameworks, neuroimaging analysis using SPM and Nipype, and detailed statistical learning recovery procedures.

## Data Access

### Behavioral Data
Behavioral experimental data are available in `./data/behavior/` and include trial-by-trial responses, reaction times, and stimulus parameters for all subjects.

### fMRI Data
fMRI data and full BIDS-formatted dataset are publicly available on [OpenNeuro](https://openneuro.org/). This includes:
- Raw fMRI BOLD timeseries
- Preprocessed fMRI data (via fMRIprep)
- Anatomical images (defaced for privacy)
- Behavioral event files
- Eye-tracking data

### Full Project Data
Complete project datasets and code are stored on the science cluster:  
`/shares/zne.uzh/multlearn/`

## Project Structure

### 📁 **fMRI/** – Experimental Task & Data Preprocessing

**Experimental Task (`fMRI/Psychtoolbox`)**
- **`multisensoryLearningTask.m`** – Main MATLAB/Psychtoolbox task implementation for fMRI
- **`taskRunWise.m`** – Run-wise task structure and trial organization
- **`createTrialStructure.ipynb`** – Trial generation and stimulus pairing logic
- **`+experiment/`** – Modular experiment components:
  - `displayCues.m` – Stimulus presentation
  - `displayReward.m` / `displayFinalScreen.m` – Feedback delivery
  - `scannerTrigger.m` – fMRI pulse synchronization
  - `tactileExample.m` / `createTouch.m` / `transformTouch.m` – Tactile stimulus handling
  - `keyboardResponse.m` – Participant response collection

**Data Preprocessing**
- **`preprocess_data/`** – BIDS conversion and event file generation
  - `raw_to_bids.py` – Converts raw data to BIDS format
  - `convert_behavior.py` – Aligns behavioral logs with fMRI timeseries and creates BIDS event files
  - `bold_template.json` – BIDS template for fMRI scans
- **`preprocess_cluster/`** – Cluster job submission scripts
  - `fmriprep.sh` – fMRIprep preprocessing pipeline
  - `deface.py` / `submit_deface.sh` – Anatomical defacing for privacy

**Analysis**
- **`analyseEyetracking.m`** – Eye-tracking data processing and validation
- **`checkAudio.m`** / **`checkTactile.m`** – Stimulus delivery validation

### 📁 **Modelling/** – Computational Modeling & Parameter Recovery

**Task Design & Simulation**
- **`TaskDesign.py`** – Core task simulation with multiple learning model variants
  - Implements trial-by-trial stimulus presentation with audiovisual/visuotactile pairings
  - Supports multiple RL variants (standard, dynamic, asymmetric, transfer learning, Pearce-Hall)
  - Three position options (3x3 grid) with learnable reward probabilities (80% reward rate)
- **`TaskDesignSingle.py`** – Single stimulus pair variant
- **`taskSimulation.py`** – Full task simulation for generating synthetic data

**Reinforcement Learning Model Fitting**
- **`Fitting/RLparameterFitting.py`** – Main parameter optimization pipeline
  - Grid search over RL parameters
  - Implements model variants:
    - Separate learning rates (α) for positive and negative prediction errors
    - Dynamic learning rate models
    - Pearce-Hall attentional learning
    - Transfer learning across stimulus pairs
    - Asymmetric learning
  - Model specifications in `BestFitting*.tsv` files
- **`Fitting/RLparameterPlotting.py`** – Visualization of fitted parameters
- **`Fitting/BIC.py`** – Bayesian Information Criterion model comparison

**Parameter Recovery & Validation**
- **`Recovery/modelRecovery.py`** – Validates that models can recover known parameters
- **`Recovery/parameterRecovery.py`** – Parameter recovery simulations
- **`Recovery/check_arrays.py`** – Validation utilities
- **`Recovery/plotting.py`** – Recovery analysis visualizations

### 📁 **SPM/** – fMRI Statistical Analysis

**GLM Analysis (Nipype-based)**
- **`GLM/glm.sh`** – Master GLM script
- **`cluster/glm_1stlevel.sh`** – First-level individual subject GLMs
- **`cluster/submit_GLM_2ndlevel_nipype.sh`** – Second-level group analysis
- Includes event regressors for:
  - Choice presentation
  - Feedback delivery
  - Run-specific effects

**ROI & PPI Analysis**
- **`cluster/extract_roi_ts.sh`** – Extract time series from regions of interest
- **`cluster/glm_1stlevel_ppi.sh`** – First-level PPI (psychophysiological interaction) GLM
- **`cluster/2nd_level_ppi.sh`** – Second-level PPI analysis
- **`cluster/submit_extractBetas.sh`** – Extract parameter estimates

**Connectivity Analysis**
- **`conn_project01.mat`** – CONN toolbox connectivity project

### 📁 **DataAnalysis/** – Visualization & Statistical Analysis

- **`accuracy_plots.ipynb`** – Learning curve visualization and accuracy analysis
- **`beta_correlations.ipynb`** – Correlate fMRI activations with behavioral measures
- **`parameterDistributions.ipynb`** – Visualize fitted RL parameter distributions
- **`paper.ipynb`** – Main results and figure generation for manuscript
- **`pycortex.ipynb`** – Brain surface visualization
- **`regressorCorr.m`** – Regressor correlation analysis

### 📁 **multlearn/** – Utility Package

- **`utils/data.py`** – Data loading and preprocessing utilities
  - Subject ID management (N=62 after exclusions)
  - Behavioral data loading from BIDS format
  - Helper functions for multi-run analyses

## RL Model Parameters

The fitted RL models include the following parameters:

- **α+ (alphaPos)** – Learning rate for positive prediction errors (0-1 range)
- **α- (alphaNeg)** – Learning rate for negative prediction errors (0-1 range)
- **α2+, α2-** – Optional separate learning rates for second value representation (if `extra=True`)
- **β (beta)** – Inverse temperature/decision temperature (0-14 range)
- **K1, K2, K3, K4** – Transfer parameters for shared stimuli (if `transfer=True`)

## Experimental Design

### Task Overview
- **6 runs per session** × 60 trials per run
- **3 modality pairs (positions)** with different reward probabilities
  - Modality 0: 50% / 35% / 15% reward probability
  - Modality 1: 15% / 50% / 35% reward probability
  - Modality 2: 35% / 15% / 50% reward probability
- **Two sensory conditions** (audiovisual vs. visuotactile, counterbalanced)
- **Concurrent measurements:** Eye-tracking, reaction times, choices

### Stimuli
- **Visual:** Colored patterns and gradients
- **Auditory:** Distinct tones and beeps
- **Tactile:** Computer-controlled haptic feedback (if applicable)

## Installation & Dependencies

### Python Dependencies
```bash
pip install -e .
# Or manually install requirements:
# numpy, scipy, pandas, matplotlib, seaborn, scikit-learn
```

### MATLAB/SPM Requirements (for fMRI analysis)
- MATLAB 2020b or later
- SPM12 or SPM13
- Psychtoolbox-3 (for task presentation)
- fMRIprep (for preprocessing)
- Nipype (for pipeline automation)

### Cluster Setup
- HPC cluster with job submission capability (tested on Euler@ETH)
- Bash shell scripts for batch job submission

## Analysis Pipeline

### Step 1: Behavioral Data Processing
```bash
cd fMRI/preprocess_data/
python convert_behavior.py
```
Converts raw behavioral logs to BIDS-formatted event files.

### Step 2: fMRI Preprocessing
```bash
cd fMRI/preprocess_cluster/
sbatch submit_deface.sh  # Anonymize anatomicals
sbatch fmriprep.sh       # Run fMRIprep
```

### Step 3: RL Model Fitting
```bash
cd Modelling/Fitting/
python RLparameterFitting.py --subject <ID> --gridCount 5000 --method valid
```

### Step 4: Parameter Recovery
```bash
cd Modelling/Recovery/
python parameterRecovery.py
python plotting.py
```

### Step 5: GLM Analysis
```bash
cd SPM/cluster/
sbatch submit_glm_1st.sh      # First-level GLMs
sbatch submit_GLM_2ndlevel.sh # Second-level analysis
```

### Step 6: PPI Analysis
```bash
sbatch submit_2ndlevel_ppi.sh
```

### Step 7: ROI Analysis
```bash
sbatch submit_extract_roi_regressors.sh
```

## Visualization & Results

All main figures and analyses are generated in the Jupyter notebooks:

- **Accuracy across trials** – How learning progresses
- **Parameter distributions** – Individual differences in learning strategies
- **Brain activation maps** – Group-level fMRI results
- **Connectivity analysis** – Functional connectivity changes
- **Behavioral correlations** – Link between learning parameters and brain activity

## File Organization Notes

- **Subjects:** 62 participants (IDs 1-64, excluding 31 and 32)
- **Runs:** 6 fMRI runs per subject
- **Trials:** 60 trials per run (360 total per subject)
- **Event types:** Choice (stimulus), Feedback, Run-level effects

## Contact & Support

For technical questions, data access, or methodological inquiries:

- **Ella Casimiro** – ella.casimiro@econ.uzh.ch
- **Saurabh Bedi** – saurabh.bedi@econ.uzh.ch
- **Gilles de Hollander** – gilles.de.hollander@gmail.com

## Citation

If you use code or data from this project, please cite:

*[Citation to be added upon publication]*

## License

[To be specified - please update as appropriate]
