function func_contrast_level1_Pmod(path, sub, model_version, del_old_con, splitting)

% path of data
data.source = [fullfile(path.folder_processed , sub)];
%path of results
if ischar(splitting)
    data.destination = fullfile(path.SPM_folder,'/results', splitting, model_version, sub);
else
    data.destination = fullfile(path.SPM_folder,'/results', model_version, sub);
end
spm_jobman('initcfg');

load(fullfile(data.destination,'SPM.mat'));
%%
runs_present = 6;

var_int = model_version;
num_pmods = find(contains(SPM.xX.name,{['x' var_int 'Tactile^' ], ['x' var_int 'Audio^']}));
% check whether the specification is correct
if numel(num_pmods) ~= runs_present
   error('Error occurred. Pmods not correct!')
end



%% contrasts

contrasts = double(contains(SPM.xX.name,{['x' var_int 'Tactile^' ], ['x' var_int 'Audio^']}));
contrasts_audio = double(contains(SPM.xX.name,['x' var_int 'Audio^']));
contrasts_tactile = double(contains(SPM.xX.name,['x' var_int 'Tactile^' ]));


%% SPE/RPE

matlabbatch{1}.spm.stats.con.spmmat = cellstr(fullfile(data.destination, 'SPM.mat'));
matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = strcat([model_version '_pos']);
matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = contrasts;
matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = strcat([model_version '_audio']);
matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = contrasts_audio;
matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{3}.tcon.name = strcat([model_version '_tactile']);
matlabbatch{1}.spm.stats.con.consess{3}.tcon.weights = contrasts_tactile;
matlabbatch{1}.spm.stats.con.consess{3}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{4}.tcon.name = strcat([model_version '_tactile-audio']);
matlabbatch{1}.spm.stats.con.consess{4}.tcon.weights = contrasts_tactile - contrasts_audio;
matlabbatch{1}.spm.stats.con.consess{4}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{5}.tcon.name = strcat([model_version '_audio-tactile']);
matlabbatch{1}.spm.stats.con.consess{5}.tcon.weights = contrasts_audio - contrasts_tactile;
matlabbatch{1}.spm.stats.con.consess{5}.tcon.sessrep = 'none';


matlabbatch{1}.spm.stats.con.delete = del_old_con;
%%
spm('defaults', 'fMRI');
spm_jobman('run', matlabbatch);
clear matlabbatch

end



