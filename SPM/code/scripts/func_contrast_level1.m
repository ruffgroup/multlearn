function func_contrast_level1(path, sub, del_old_con)

% path of data
data.source = [fullfile(path.folder_processed , sub)];
%path of results
data.destination = fullfile(path.SPM_folder,'/results/other', sub);

spm_jobman('initcfg');

load(fullfile(data.destination,'SPM.mat'));

%% contrasts
if numel(find(contains(SPM.xX.name,['Sn(1) ChoiceAudio']))) > 0
    contrasts_audio = double(contains(SPM.xX.name, {['Sn(1) constant'], ['Sn(3) constant'], ['Sn(5) constant']}));
    contrasts_tactile = double(contains(SPM.xX.name, {['Sn(2) constant'], ['Sn(4) constant'], ['Sn(6) constant']}));
elseif numel(find(contains(SPM.xX.name,['Sn(1) ChoiceTactile']))) > 0
    contrasts_tactile = double(contains(SPM.xX.name, {['Sn(1) constant'], ['Sn(3) constant'], ['Sn(5) constant']}));
    contrasts_audio = double(contains(SPM.xX.name, {['Sn(2) constant'], ['Sn(4) constant'], ['Sn(6) constant']}));
else
    error("Constants audio and tactile not defined")
end

matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'tactile-audio';
matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = contrasts_tactile - contrasts_audio;
matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = 'audio-tactile';
matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = contrasts_audio - contrasts_tactile;
matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none';


matlabbatch{1}.spm.stats.con.delete = del_old_con;
%%
spm('defaults', 'fMRI');
spm_jobman('run', matlabbatch);
clear matlabbatch

end
