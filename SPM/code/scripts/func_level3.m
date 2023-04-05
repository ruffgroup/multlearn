function func_level3(path, model_version)


% path of data
data_files = dir(fullfile(path.project, 'results', model_version, ['Second_level_' model_version '_con*' ] ));


spm_jobman('initcfg');


%% Global positive

matlabbatch{1}.spm.stats.con.spmmat = {[data_files(1).folder, filesep, data_files(1).name, filesep ,'SPM.mat']};
matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = strcat(model_version,'global_pos');
matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = 1;
matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
matlabbatch{1}.spm.stats.con.delete = 1;

matlabbatch{2}.spm.stats.con.spmmat = {[data_files(1).folder, filesep, data_files(2).name, filesep ,'SPM.mat']};
matlabbatch{2}.spm.stats.con.consess{1}.tcon.name = strcat(model_version,'level_1');
matlabbatch{2}.spm.stats.con.consess{1}.tcon.weights = 1;
matlabbatch{2}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
matlabbatch{2}.spm.stats.con.delete = 1;

matlabbatch{3}.spm.stats.con.spmmat = {[data_files(1).folder, filesep, data_files(3).name, filesep ,'SPM.mat']};
matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = strcat(model_version,'level_2');
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;
matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
matlabbatch{3}.spm.stats.con.delete = 1;

matlabbatch{4}.spm.stats.con.spmmat = {[data_files(1).folder, filesep, data_files(4).name, filesep ,'SPM.mat']};
matlabbatch{4}.spm.stats.con.consess{1}.tcon.name = strcat(model_version,'level_3');
matlabbatch{4}.spm.stats.con.consess{1}.tcon.weights = 1;
matlabbatch{4}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
matlabbatch{4}.spm.stats.con.delete = 1;

matlabbatch{5}.spm.stats.con.spmmat = {[data_files(1).folder, filesep, data_files(5).name, filesep ,'SPM.mat']};
matlabbatch{5}.spm.stats.con.consess{1}.tcon.name = strcat(model_version,'lev2-lev1');
matlabbatch{5}.spm.stats.con.consess{1}.tcon.weights = 1;
matlabbatch{5}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
matlabbatch{5}.spm.stats.con.delete = 1;

matlabbatch{6}.spm.stats.con.spmmat = {[data_files(1).folder, filesep, data_files(6).name, filesep ,'SPM.mat']};
matlabbatch{6}.spm.stats.con.consess{1}.tcon.name = strcat(model_version,'lev3-lev1');
matlabbatch{6}.spm.stats.con.consess{1}.tcon.weights = 1;
matlabbatch{6}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
matlabbatch{6}.spm.stats.con.delete = 1;

matlabbatch{7}.spm.stats.con.spmmat = {[data_files(1).folder, filesep, data_files(7).name, filesep ,'SPM.mat']};
matlabbatch{7}.spm.stats.con.consess{1}.tcon.name = strcat(model_version,'lev3-lev2');
matlabbatch{7}.spm.stats.con.consess{1}.tcon.weights = 1;
matlabbatch{7}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
matlabbatch{7}.spm.stats.con.delete = 1;



%%
spm('defaults', 'fMRI');
spm_jobman('run', matlabbatch);
clear matlabbatch



%% run the analysis
% 
% for i = 1:numel(data_files)
%     matlabbatch{i}.spm.stats.fmri_est.spmmat = {[data_files(1).folder, filesep, data_files(i).name, filesep ,'SPM.mat']};
%     matlabbatch{i}.spm.stats.fmri_est.write_residuals = 0;
%     matlabbatch{i}.spm.stats.fmri_est.method.Classical = 1;   
% end
% 
% spm('defaults', 'fMRI');
% spm_jobman('run', matlabbatch);
% clear matlabbatch



end