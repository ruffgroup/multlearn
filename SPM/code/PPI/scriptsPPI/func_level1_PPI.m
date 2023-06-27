function func_level1_PPI(folder_processed, bids_folder, SPM_folder, sub, model_version, ROI_folder, ROI)

spm('defaults', 'fMRI');

% assume that there is
filter_thr = 128; % because we are filtering with fMRIPrep already. otherwise 180 or 200 is good.
param_smoothing = 6; % what was the smoothing parameter?

ROI_name = split(ROI.name,'_');
ROI_name = strcat(ROI_name(1,1), '_',ROI_name(2,1));

%path of results
data.destination = string(fullfile(SPM_folder,'/results',model_version,'PPI', ROI_folder, ROI_name, sub));

%% start spm 1stLvL model
if ~exist(data.destination,'dir')
    mkdir(data.destination)
end

data.destination = dir([data.destination]);
data.destination = data.destination(end).folder;

spm_jobman('initcfg');

%% Get the specific runs (with number)
data.source = dir(fullfile(folder_processed,sub, '/func', [ 's' num2str(param_smoothing) '.' sub '_*_run-*_bold.nii']));
% get the run number from the file name
run_index = (regexp([data.source(:).name], '(?<=_run-)[0-9]', 'match'))';
run_name = [run_index {data.source.name}'];

[sortedValues, sortOrder] = sort(run_name(:,1));
run_name = run_name(sortOrder, 2);

%% fMRI model specification

matlabbatch{1}.spm.stats.fmri_spec.dir = {data.destination};
matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';

%%
motion_param = [];
other_param = [];
R = [];
physio = [];
names = [];
if ~exist(fullfile(folder_processed,sub, ['beh'],['nuisance_' sub '_PPI.mat']))

    for nrun = 1:numel(data.source)
    % LOAD physio data
    physioR_file = (fullfile(folder_processed,sub, ['beh/physio/RegPhysio_' sub '_run_' num2str(nrun) '.mat']));
    if ~exist(physioR_file, 'file')
        physio.model.R_column_names = [];
        physio.model.R = [];      
    else
        load(fullfile(folder_processed,sub, ['beh/physio/RegPhysio_' sub '_run_' num2str(nrun) '.mat']));
    end
    
    %% Load motion regressors
    
    file_confounds = dir(fullfile(folder_processed,sub, ['func/' sub '*_run-' num2str(nrun) '*confounds_timeseries.tsv']));
    confounds_raw = tdfread(fullfile(file_confounds.folder,file_confounds.name));
    
    % Regress out the 6 std motion params and the three low frequency noise parameters
    motion_param = [confounds_raw.trans_x confounds_raw.trans_y confounds_raw.trans_z confounds_raw.rot_x confounds_raw.rot_y confounds_raw.rot_z];
    %str2double(strtrim(string(confounds_raw.dvars))) str2double(strtrim(string(confounds_raw.framewise_displacement)))
    other_param = [confounds_raw.a_comp_cor_00 confounds_raw.a_comp_cor_01 confounds_raw.a_comp_cor_02 confounds_raw.a_comp_cor_03 confounds_raw.a_comp_cor_04];
    
    %% Combine everything to nuisance regressors
    add_motion_names = [];
    for ii = 1:numel(motion_param(1,:))
        add_motion_names = [add_motion_names; 'MotionReg_',num2str(ii)];
    end
    add_motion_names =  cellstr(add_motion_names)';
    
    add_other_names = [];
    for ii = 1:numel(other_param(1,:))
        add_other_names = [add_other_names; 'aCompCor_', num2str(ii)];
    end
    add_other_names = cellstr(add_other_names)';
    
    if isempty(names)
    names = [physio.model.R_column_names add_motion_names add_other_names]  ;
    end

    R = [R;physio.model.R motion_param other_param] ;
    %%
    
    end
save(fullfile(folder_processed,sub, ['beh'],['nuisance_' sub '_PPI.mat']),'names', 'R');
else
    file_nuisance = dir([fullfile(folder_processed,sub, ['beh'],['nuisance_' sub '_PPI.mat'])]);
end



% FIND CONSTANTS
info_scan = dir(strcat(bids_folder, sub, '/func/', sub, '_*_run-' ,num2str(1), '_bold.json'));
info_scan = read_json(fullfile(info_scan(end).folder, info_scan(end).name));
TR = info_scan.RepetitionTime;
Nslices = info_scan.MaxSlices;
refSlice = round(Nslices/2); % IMPORTANT: CHANGE THIS VALUE DEPENDING ON THE REFERENCE SLICE

%% SCAN PARAMS
matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TR;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = Nslices;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = refSlice;

%% SCANS
allScans = [];
for nrun = 1:numel(data.source)
    allScans = [allScans;fullfile(data.source(nrun).folder, run_name{nrun})];
end
matlabbatch{1}.spm.stats.fmri_spec.sess(1).scans = cellstr([spm_select('expand',allScans)]);
%%

matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});

multicondition_file=fullfile(folder_processed, sub,['beh'],model_version,filesep,['PPI_conditions.mat']);
matlabbatch{1}.spm.stats.fmri_spec.sess(1).multi = cellstr(string(multicondition_file));
%matlabbatch{1}.spm.stats.fmri_spec.sess(nrun).multi = {cellstr(fullfile(folder_processed, sub,['beh'],model_version, ['run_' num2str(nrun) '_conditions.mat']))};
matlabbatch{1}.spm.stats.fmri_spec.sess(1).regress = struct('name', {}, 'val', {});
matlabbatch{1}.spm.stats.fmri_spec.sess(1).multi_reg = {[fullfile(file_nuisance.folder,file_nuisance.name)]}; %note there should be always one file here
matlabbatch{1}.spm.stats.fmri_spec.sess(1).hpf = filter_thr;


matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
matlabbatch{1}.spm.stats.fmri_spec.mask = {strcat(ROI.folder,filesep, ROI.name, ',1')};
matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

%% fMRI model estimation

%matlabbatch{2}.spm.stats.review.spmmat =  {[fullfile(data.destination, 'SPM.mat')]};
%matlabbatch{2}.spm.stats.review.display.orth = 1;
%matlabbatch{2}.spm.stats.review.print = 'png';
matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('fMRI model specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

%% run batch
tic

spm_jobman('run', matlabbatch);
clear matlabbatch

toc

%%

end
