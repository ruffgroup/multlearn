function func_contrast_level1_PPI(path, sub, model_version, del_old_con, splitting, ROI_folder, ROI)

%path of results
ROI_name = split(ROI.name,'_');
ROI_name = strcat(ROI_name(1,1), '_',ROI_name(2,1));

data.destination = fullfile(path.SPM_folder,'/results', splitting, 'PPI', ROI_folder, ROI_name, sub);

spm_jobman('initcfg');
if isfile(fullfile(data.destination,'SPM.mat'))
load(string(fullfile(data.destination,'SPM.mat')));
%%
runs_present = 6;

var_int = model_version;
num_pmods = find(contains(SPM.xX.name,'PPI'));
% check whether the specification is correct
if numel(num_pmods) ~= runs_present
   error('Error occurred. Pmods not correct!')
end



%% contrasts

contrastsPPI = double(contains(SPM.xX.name,'PPI'));
if numel(find(contains(SPM.xX.name,['Sn(1) ChoiceAudio']))) > 0
    contrastsAudioPPI = double(contains(SPM.xX.name, {['Sn(1) PPI'], ['Sn(3) PPI'], ['Sn(5) PPI']}));
    contrastsTactilePPI = double(contains(SPM.xX.name, {['Sn(2) PPI'], ['Sn(4) PPI'], ['Sn(6) PPI']}));
elseif numel(find(contains(SPM.xX.name,['Sn(1) ChoiceTactile']))) > 0
    contrastsTactilePPI = double(contains(SPM.xX.name, {['Sn(1) PPI'], ['Sn(3) PPI'], ['Sn(5) PPI']}));
    contrastsAudioPPI = double(contains(SPM.xX.name, {['Sn(2) PPI'], ['Sn(4) PPI'], ['Sn(6) PPI']}));
else
    error("Constants audio and tactile not defined")
end


%% SPE/RPE

matlabbatch{1}.spm.stats.con.spmmat = cellstr(fullfile(data.destination, 'SPM.mat'));
matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'Interaction';
matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = contrastsPPI;
matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = 'tactile-audio';
matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = contrastsTactilePPI - contrastsAudioPPI;
matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{3}.tcon.name = 'audio-tactile';
matlabbatch{1}.spm.stats.con.consess{3}.tcon.weights = contrastsAudioPPI - contrastsTactilePPI;
matlabbatch{1}.spm.stats.con.consess{3}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{4}.tcon.name = 'tactile';
matlabbatch{1}.spm.stats.con.consess{4}.tcon.weights = contrastsTactilePPI;
matlabbatch{1}.spm.stats.con.consess{4}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{5}.tcon.name = 'audio';
matlabbatch{1}.spm.stats.con.consess{5}.tcon.weights = contrastsAudioPPI;
matlabbatch{1}.spm.stats.con.consess{5}.tcon.sessrep = 'none';




matlabbatch{1}.spm.stats.con.delete = del_old_con;
%%
spm('defaults', 'fMRI');
spm_jobman('run', matlabbatch);
clear matlabbatch
end
end



