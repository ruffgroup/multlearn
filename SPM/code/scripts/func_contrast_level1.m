function func_contrast_level1(path, sub, model_version, del_old_con)

% path of data
data.source = [fullfile(path.folder_processed , sub)];
%path of results
data.destination = fullfile(path.SPM_folder,'/results', model_version, sub);

spm_jobman('initcfg');

load(fullfile(data.destination,'SPM.mat'));
%%
runs_present = 6;
if model_version == "SPE"
    var_int = 'Statistical';

end
num_pmods = find(contains({SPM.xX.name},{['x' var_int 'Tactile^' ], ['x' var_int 'Audio^']}));
% check whether the specification is correct
if numel(num_pmods) ~= runs_present
    error('Error occurred. Variable of interest not correct!')
end


%% globally

contrasts_global = double(contains({SPM.xX.name},{['x' var_int 'Tactile^' ], ['x' var_int 'Audio^']}));
contrasts_audio = double(contains(SPM.xX.name,['x' var_int 'Audio^']));
contrasts_tactile = double(contains(SPM.xX.name,['x' var_int 'Tactile^' ]));


%% SPE

matlabbatch{1}.spm.stats.con.spmmat = cellstr(fullfile(data.destination, 'SPM.mat'));
matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'SPE_pos';
matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = contrasts_global;
matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.delete = del_old_con;

matlabbatch{2}.spm.stats.con.consess{1}.tcon.name = 'SPE_audio';
matlabbatch{2}.spm.stats.con.consess{1}.tcon.weights = contrasts_audio;
matlabbatch{2}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

matlabbatch{2}.spm.stats.con.delete = del_old_con;

matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'SPE_tactile';
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = contrasts_tactile;
matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

matlabbatch{3}.spm.stats.con.delete = del_old_con;
%%
spm('defaults', 'fMRI');
spm_jobman('run', matlabbatch);
clear matlabbatch

end



