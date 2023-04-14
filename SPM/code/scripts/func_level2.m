<<<<<<< HEAD
function func_level2(path,subs, model_version, contrast_num)

% path of data
master_folder = ['D:/multlearn/SPM'];
data.destination = fullfile(master_folder, 'results', model_version, ['Second_level_' model_version '_con' num2str(contrast_num )] );

spm_jobman('initcfg');

folder_files = dir(fullfile(master_folder, 'results', model_version, 'sub-*'));

for ii = 1:numel(subs)
    contrasts{ii,1} = ([ folder_files(ii).folder, filesep, subs(ii).name , filesep, 'con_000' , num2str(contrast_num) ,'.nii,1' ]);
end

matlabbatch{1}.spm.stats.factorial_design.dir = {data.destination};
matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = contrasts;
matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

%% run the analysis
matlabbatch{2}.spm.stats.fmri_est.spmmat = {strcat(data.destination, '/SPM.mat')};
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
spm('defaults', 'fMRI');
spm_jobman('run', matlabbatch);
clear matlabbatch

=======
function func_level2(path,subs, model_version, contrast_num)

% path of data
data.destination = fullfile(path.SPM_folder,'/results', model_version, strcat('Second_level_con', num2str(contrast_num )), filesep );

spm_jobman('initcfg');

folder_files = dir(fullfile(path.SPM_folder,'/results', model_version, 'sub-*'));

for ii = 1:numel(subs)
    contrasts{ii,1} = ([ folder_files(ii).folder, filesep, subs(ii).name , filesep, 'con_000' , num2str(contrast_num) ,'.nii,1' ]);
end

matlabbatch{1}.spm.stats.factorial_design.dir = cellstr(string(data.destination));
matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = contrasts;
matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

%% run the analysis
matlabbatch{2}.spm.stats.fmri_est.spmmat = cellstr(string(fullfile(data.destination, 'SPM.mat')));
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
spm('defaults', 'fMRI');
spm_jobman('run', matlabbatch);
clear matlabbatch

>>>>>>> 5f4c3248c8658a48e33ddde3a64a3c3023bbd04f
end