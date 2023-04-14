<<<<<<< HEAD
function func_smooth(folder, sub, nruns)

spm('defaults', 'FMRI');
% memory (needs to be changed according to computer)
fhwm = 6; % setzte den smoothing kernel
%%

files = dir(strcat(folder,'/',sub, '/func/', sub, '_*_run-*_bold.nii.gz'));

if numel(files) > 0
    for ii = 1:numel(files)
        gunzip(fullfile(files(ii).folder, files(ii).name));
    end
end

WaitSecs(.1)

files = dir(strcat(folder,'/',sub, '/func/', sub, '_*_run-*MNI152NLin2009cAsym_desc-preproc_bold.nii'));
if numel(files) > 0
    while numel(files) ~= nruns
        WaitSecs(1)
        files = dir(strcat(folder,'/',sub, '/func/', sub, '_*_run-*MNI152NLin2009cAsym_desc-preproc_bold.nii'));
    end
end
WaitSecs(1)

%%
tic
for ii = 1:numel(files)

    if ~exist(fullfile(files(ii).folder, [ 's' num2str(fhwm) '.' files(ii).name ]))

        WaitSecs(3)
        matlabbatch{1}.spm.spatial.smooth.data = cellstr([spm_select('expand',[fullfile(files(ii).folder, files(ii).name)])]);

        matlabbatch{1}.spm.spatial.smooth.fwhm = [fhwm fhwm fhwm];
        matlabbatch{1}.spm.spatial.smooth.dtype = 0;
        matlabbatch{1}.spm.spatial.smooth.im = 0;
        matlabbatch{1}.spm.spatial.smooth.prefix = ['s' num2str(fhwm) '.'];

        spm_jobman('run', matlabbatch);
        clear matlabbatch

    end


end
toc
%%
=======
function func_smooth(folder, sub, nruns)

spm('defaults', 'FMRI');
% memory (needs to be changed according to computer)
fhwm = 6; % setzte den smoothing kernel
%%

files = dir(strcat(folder,'/',sub, '/func/', sub, '_*_run-*_bold.nii.gz'));

if numel(files) > 0
    for ii = 1:numel(files)
        gunzip(fullfile(files(ii).folder, files(ii).name));
    end
end

WaitSecs(.1)

files = dir(strcat(folder,'/',sub, '/func/', sub, '_*_run-*MNI152NLin2009cAsym_desc-preproc_bold.nii'));
if numel(files) > 0
    while numel(files) ~= nruns
        WaitSecs(1)
        files = dir(strcat(folder,'/',sub, '/func/', sub, '_*_run-*MNI152NLin2009cAsym_desc-preproc_bold.nii'));
    end
end
WaitSecs(1)

%%
tic
for ii = 1:numel(files)

    if ~exist(fullfile(files(ii).folder, [ 's' num2str(fhwm) '.' files(ii).name ]))

        matlabbatch{1}.spm.spatial.smooth.data = cellstr([spm_select('expand',[fullfile(files(ii).folder, files(ii).name)])]);

        matlabbatch{1}.spm.spatial.smooth.fwhm = [fhwm fhwm fhwm];
        matlabbatch{1}.spm.spatial.smooth.dtype = 0;
        matlabbatch{1}.spm.spatial.smooth.im = 0;
        matlabbatch{1}.spm.spatial.smooth.prefix = ['s' num2str(fhwm) '.'];

        spm_jobman('run', matlabbatch);
        clear matlabbatch

    end


end
toc
%%
>>>>>>> 5f4c3248c8658a48e33ddde3a64a3c3023bbd04f
end