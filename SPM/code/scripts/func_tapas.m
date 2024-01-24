function physio = func_tapas(sub, nrun, folder_processed, bids_folder)

output_dir = [strcat(folder_processed, sub, '/beh/physio')];
input_log = dir(strcat(bids_folder, sub, '/func/', sub, '_*_run-' ,num2str(nrun), '_physio.log'));

if numel(input_log) ~= 0
    input_log = fullfile(input_log(end).folder,input_log(end).name);

    info_scan = dir(strcat(bids_folder, sub, '/func/', sub, '_*_run-' ,num2str(nrun), '_bold.json'));
    info_scan = read_json(fullfile(info_scan(end).folder,info_scan(end).name));

    n_slices = info_scan.MaxSlices;
    TR = info_scan.RepetitionTime;
    Ndummies = 5; % default from Karl
    Nscans = info_scan.MaxDynamics;
    onset_slice = 20; %NOTE: after pre-processing this is the middle slice


    %% Create default parameter structure with all fields
    physio = tapas_physio_new();

    %% -----------------------------------------------------------------------
    physio.save_dir = {output_dir};
    physio.log_files.vendor = 'Philips';
    physio.log_files.cardiac = {input_log};
    physio.log_files.respiration = {input_log};
    physio.log_files.scan_timing = {input_log};
    physio.log_files.sampling_interval = [];
    physio.log_files.relative_start_acquisition = 0;
    physio.log_files.align_scan = 'last';
    physio.scan_timing.sqpar.Nslices = n_slices;
    physio.scan_timing.sqpar.NslicesPerBeat = [];
    physio.scan_timing.sqpar.TR = TR;
    physio.scan_timing.sqpar.Ndummies = Ndummies;
    physio.scan_timing.sqpar.Nscans = Nscans;
    physio.scan_timing.sqpar.onset_slice = 20;
    physio.scan_timing.sqpar.time_slice_to_slice = [];
    physio.scan_timing.sqpar.Nprep = [];
    physio.scan_timing.sync.method = 'nominal';
    physio.scan_timing.sync.nominal = struct([]);
    physio.preproc.cardiac.modality = 'ECG';
    physio.preproc.cardiac.filter.no = struct([]);
    physio.preproc.cardiac.initial_cpulse_select.auto_matched.min = 0.4;
    physio.preproc.cardiac.initial_cpulse_select.auto_matched.file = 'initial_cpulse_kRpeakfile.mat';
    physio.preproc.cardiac.initial_cpulse_select.auto_matched.max_heart_rate_bpm = 150;
    physio.preproc.cardiac.initial_cpulse_select.max_heart_rate_bpm = 150;
    physio.preproc.cardiac.posthoc_cpulse_select.off = struct([]);
    physio.preproc.respiratory.filter.passband = [0.01 2];
    physio.preproc.respiratory.despike = false;
    physio.model.output_multiple_regressors = ['RegPhysio_',sub,'_run_',num2str(nrun),'.txt'];
    physio.model.output_physio = ['RegPhysio_',sub,'_run_',num2str(nrun),'.mat'];
    physio.model.orthogonalise = 'none';
    physio.model.censor_unreliable_recording_intervals = true;
    physio.model.retroicor.yes.order.c = 3;
    physio.model.retroicor.yes.order.r = 4;
    physio.model.retroicor.yes.order.cr = 1;
    physio.model.rvt.no = struct([]);
    physio.model.hrv.no = struct([]);
    physio.model.noise_rois.no = struct([]);
    physio.model.movement.no = struct([]);
    physio.model.other.no = struct([]);
    physio.verbose.level = 1;
    physio.verbose.fig_output_file = ['fig_',sub,'_run_',num2str(nrun),'.png'];
    physio.verbose.use_tabs = false;


    %% run job

    physio = tapas_physio_main_create_regressors(physio);
end



end
=======
function physio = func_tapas(sub, nrun, folder_processed, bids_folder)

output_dir = [strcat(folder_processed, sub, '/beh/physio')];
input_log = dir(strcat(bids_folder, sub, '/func/', sub, '_*_run-' ,num2str(nrun), '_physio.log'));

if numel(input_log) ~= 0
    input_log = fullfile(input_log(end).folder,input_log(end).name);

    info_scan = dir(strcat(bids_folder, sub, '/func/', sub, '_*_run-' ,num2str(nrun), '_bold.json'));
    info_scan = read_json(fullfile(info_scan(end).folder,info_scan(end).name));

    n_slices = info_scan.MaxSlices;
    TR = info_scan.RepetitionTime;
    Ndummies = 5; % default from Karl
    Nscans = info_scan.MaxDynamics;
    onset_slice = round(n_slices/2); %NOTE: after pre-processing this is the middle slice


    %% Create default parameter structure with all fields
    physio = tapas_physio_new();

    %% -----------------------------------------------------------------------
    physio.save_dir = {output_dir};
    physio.log_files.vendor = 'Philips';
    physio.log_files.cardiac = {input_log};
    physio.log_files.respiration = {input_log};
    physio.log_files.scan_timing = {input_log};
    physio.log_files.sampling_interval = [];
    physio.log_files.relative_start_acquisition = 0;
    physio.log_files.align_scan = 'last';
    physio.scan_timing.sqpar.Nslices = n_slices;
    physio.scan_timing.sqpar.NslicesPerBeat = [];
    physio.scan_timing.sqpar.TR = TR;
    physio.scan_timing.sqpar.Ndummies = Ndummies;
    physio.scan_timing.sqpar.Nscans = Nscans;
    physio.scan_timing.sqpar.onset_slice = onset_slice;
    physio.scan_timing.sqpar.time_slice_to_slice = [];
    physio.scan_timing.sqpar.Nprep = [];
    physio.scan_timing.sync.method = 'nominal';
    physio.scan_timing.sync.nominal = struct([]);
    physio.preproc.cardiac.modality = 'ECG';
    physio.preproc.cardiac.filter.no = struct([]);
    physio.preproc.cardiac.initial_cpulse_select.auto_matched.min = 0.4;
    physio.preproc.cardiac.initial_cpulse_select.auto_matched.file = 'initial_cpulse_kRpeakfile.mat';
    physio.preproc.cardiac.initial_cpulse_select.auto_matched.max_heart_rate_bpm = 150;
    physio.preproc.cardiac.initial_cpulse_select.max_heart_rate_bpm = 150;
    physio.preproc.cardiac.posthoc_cpulse_select.off = struct([]);
    physio.preproc.respiratory.filter.passband = [0.01 2];
    physio.preproc.respiratory.despike = false;
    physio.model.output_multiple_regressors = ['RegPhysio_',sub,'_run_',num2str(nrun),'.txt'];
    physio.model.output_physio = ['RegPhysio_',sub,'_run_',num2str(nrun),'.mat'];
    physio.model.orthogonalise = 'none';
    physio.model.censor_unreliable_recording_intervals = true;
    physio.model.retroicor.yes.order.c = 3;
    physio.model.retroicor.yes.order.r = 4;
    physio.model.retroicor.yes.order.cr = 1;
    physio.model.rvt.no = struct([]);
    physio.model.hrv.no = struct([]);
    physio.model.noise_rois.no = struct([]);
    physio.model.movement.no = struct([]);
    physio.model.other.no = struct([]);
    physio.verbose.level = 1;
    physio.verbose.fig_output_file = ['fig_',sub,'_run_',num2str(nrun),'.png'];
    physio.verbose.use_tabs = false;


    %% run job

    physio = tapas_physio_main_create_regressors(physio);
end



end
