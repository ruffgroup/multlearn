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
num_pmods = find(contains(SPM.xX.name,{['x' var_int 'Tactile^' ], ['x' var_int 'Audio^']}));
% check whether the specification is correct
if numel(num_pmods) ~= runs_present
    error('Error occurred. Variable of interest not correct!')
end


%% contrasts

contrasts_SPE = double(contains(SPM.xX.name,{['x' var_int 'Tactile^' ], ['x' var_int 'Audio^']}));
contrasts_audioSPE = double(contains(SPM.xX.name,['x' var_int 'Audio^']));
contrasts_tactileSPE = double(contains(SPM.xX.name,['x' var_int 'Tactile^' ]));

if numel(find(contains(SPM.xX.name,['Sn(1) ChoiceAudio']))) > 0
    contrasts_audio = double(contains(SPM.xX.name, {['Sn(1) constant'], ['Sn(3) constant'], ['Sn(5) constant']}));
    contrasts_tactile = double(contains(SPM.xX.name, {['Sn(2) constant'], ['Sn(4) constant'], ['Sn(6) constant']}));
elseif numel(find(contains(SPM.xX.name,['Sn(1) ChoiceTactile']))) > 0
    contrasts_tactile = double(contains(SPM.xX.name, {['Sn(1) constant'], ['Sn(3) constant'], ['Sn(5) constant']}));
    contrasts_audio = double(contains(SPM.xX.name, {['Sn(2) constant'], ['Sn(4) constant'], ['Sn(6) constant']}));
else
    error("Constants audio and tactile not defined")
end


%% SPE

matlabbatch{1}.spm.stats.con.spmmat = cellstr(fullfile(data.destination, 'SPM.mat'));
matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'SPE_pos';
matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = contrasts_SPE;
matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = 'SPE_audio';
matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = contrasts_audioSPE;
matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{3}.tcon.name = 'SPE_tactile';
matlabbatch{1}.spm.stats.con.consess{3}.tcon.weights = contrasts_tactileSPE;
matlabbatch{1}.spm.stats.con.consess{3}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{4}.tcon.name = 'SPE_tactile-audio';
matlabbatch{1}.spm.stats.con.consess{4}.tcon.weights = contrasts_tactileSPE - contrasts_audioSPE;
matlabbatch{1}.spm.stats.con.consess{4}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{5}.tcon.name = 'SPE_audio-tactile';
matlabbatch{1}.spm.stats.con.consess{5}.tcon.weights = contrasts_audioSPE - contrasts_tactileSPE;
matlabbatch{1}.spm.stats.con.consess{5}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{6}.tcon.name = 'tactile-audio';
matlabbatch{1}.spm.stats.con.consess{6}.tcon.weights = contrasts_tactile - contrasts_audio;
matlabbatch{1}.spm.stats.con.consess{6}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{7}.tcon.name = 'audio-tactile';
matlabbatch{1}.spm.stats.con.consess{7}.tcon.weights = contrasts_audio - contrasts_tactile;
matlabbatch{1}.spm.stats.con.consess{7}.tcon.sessrep = 'none';


matlabbatch{1}.spm.stats.con.delete = del_old_con;
%%
spm('defaults', 'fMRI');
spm_jobman('run', matlabbatch);
clear matlabbatch

end



