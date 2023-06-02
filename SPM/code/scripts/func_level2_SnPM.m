function func_level2_SnPM(path,subs, model_version, contrast_num, splitting)
spm_figure('GetWin','Graphics');
% path of data
if ischar(splitting) && splitting ~= string(model_version)
    data.destination = fullfile(path.SPM_folder,'/results', splitting, model_version, strcat('Second_level_SnPM_con', num2str(contrast_num )), filesep );
elseif ischar(splitting) && splitting == string(model_version) || ~ischar(splitting)
    data.destination = fullfile(path.SPM_folder,'/results', model_version,strcat('Second_level_SnPM_con', num2str(contrast_num )), filesep );
end

spm_jobman('initcfg');

if ischar(splitting) && splitting ~= string(model_version)
    folder_files = dir(fullfile(path.SPM_folder,'/results', splitting, model_version, 'sub-*'));
elseif ischar(splitting) && splitting == string(model_version) || ~ischar(splitting)
    folder_files = dir(fullfile(path.SPM_folder,'/results', model_version, 'sub-*'));
end


for ii = 1:numel(subs)
    if contrast_num < 10
        contrasts{ii,1} = ([ folder_files(ii).folder, filesep, subs(ii).name , filesep, 'con_000' , num2str(contrast_num) ,'.nii,1' ]);
    else
        contrasts{ii,1} = ([ folder_files(ii).folder, filesep, subs(ii).name , filesep, 'con_00' , num2str(contrast_num) ,'.nii,1' ]);
    end
end

matlabbatch{1}.spm.tools.snpm.des.OneSampT.DesignName = 'MultiSub: One Sample T test on diffs/contrasts';
matlabbatch{1}.spm.tools.snpm.des.OneSampT.DesignFile = 'snpm_bch_ui_OneSampT';
matlabbatch{1}.spm.tools.snpm.des.OneSampT.dir = cellstr(string(data.destination));
matlabbatch{1}.spm.tools.snpm.des.OneSampT.P = contrasts;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.cov = struct('c', {}, 'cname', {});
matlabbatch{1}.spm.tools.snpm.des.OneSampT.nPerm = 5000;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.vFWHM = [0 0 0];
matlabbatch{1}.spm.tools.snpm.des.OneSampT.bVolm = 1;
%matlabbatch{1}.spm.tools.snpm.des.OneSampT.ST.ST_U = tVal;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.ST.ST_later = -1;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.masking.tm.tm_none = 1;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.masking.im = 1;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.masking.em = {''};
matlabbatch{1}.spm.tools.snpm.des.OneSampT.globalc.g_omit = 1;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.globalm.glonorm = 1;

matlabbatch{2}.spm.tools.snpm.cp.snpmcfg(1) = cfg_dep('MultiSub: One Sample T test on diffs/contrasts: SnPMcfg.mat configuration file', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','SnPMcfg'));

spm('defaults', 'fMRI');
spm_jobman('run', matlabbatch);
clear matlabbatch
end
