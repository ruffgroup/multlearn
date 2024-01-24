function func_contrast_level1_Pmod_ROI(path, sub, model_version, del_old_con, splitting, ROI_folder, ROI)

%path of results
ROI_name = split(ROI.name,'_');
ROI_name = strcat(ROI_name(1,1), '_',ROI_name(2,1));
if ischar(splitting)
    if ~exist(string(fullfile(path.SPM_folder,'/results', splitting,'ROI_analysis', ROI_folder, ROI_name, model_version)),'dir')
    mkdir(string(fullfile(path.SPM_folder,'/results', splitting,'ROI_analysis', ROI_folder, ROI_name, model_version)))
    copyfile(string(fullfile(path.SPM_folder,'/results', splitting,'ROI_analysis', ROI_folder, ROI_name, sub)),string(fullfile(path.SPM_folder,'/results', splitting, 'ROI_analysis', ROI_folder, ROI_name, model_version, sub)))
    end
    data.destination = fullfile(path.SPM_folder,'/results', splitting, 'ROI_analysis', ROI_folder, ROI_name, model_version, sub);
else
    data.destination = fullfile(path.SPM_folder,'/results', model_version, 'ROI_analysis', ROI_folder, ROI_name, sub);
end
spm_jobman('initcfg');

load(string(fullfile(data.destination,'SPM.mat')));
%%
runs_present = 6;

var_int = model_version;
num_pmods = find(contains(SPM.xX.name,{['x' var_int 'Tactile^' ], ['x' var_int 'Audio^']}));
% check whether the specification is correct
if numel(num_pmods) ~= runs_present
   error('Error occurred. Pmods not correct!')
end



%% contrasts

contrastsP = double(contains(SPM.xX.name,{['x' var_int 'Tactile^' ], ['x' var_int 'Audio^']}));
contrasts_audioP = double(contains(SPM.xX.name,['x' var_int 'Audio^']));
contrasts_tactileP = double(contains(SPM.xX.name,['x' var_int 'Tactile^' ]));

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


%% SPE/RPE

matlabbatch{1}.spm.stats.con.spmmat = cellstr(fullfile(data.destination, 'SPM.mat'));
matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = strcat([model_version '_pos']);
matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = contrastsP;
matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = strcat([model_version '_audio']);
matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = contrasts_audioP;
matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{3}.tcon.name = strcat([model_version '_tactile']);
matlabbatch{1}.spm.stats.con.consess{3}.tcon.weights = contrasts_tactileP;
matlabbatch{1}.spm.stats.con.consess{3}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{4}.tcon.name = strcat([model_version '_tactile-audio']);
matlabbatch{1}.spm.stats.con.consess{4}.tcon.weights = contrasts_tactileP - contrasts_audioP;
matlabbatch{1}.spm.stats.con.consess{4}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{5}.tcon.name = strcat([model_version '_audio-tactile']);
matlabbatch{1}.spm.stats.con.consess{5}.tcon.weights = contrasts_audioP - contrasts_tactileP;
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



