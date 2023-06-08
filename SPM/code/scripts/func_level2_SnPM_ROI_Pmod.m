function func_level2_SnPM_ROI_Pmod(path,subs, model_version, contrast_num, splitting, ROI_folder, ROI)
spm_figure('GetWin','Graphics');
% path of data
ROI_name = split(ROI.name,'_');
ROI_name = strcat(ROI_name(1,1), '_',ROI_name(2,1));
data.destination = fullfile(path.SPM_folder,'/results', splitting, 'ROI_analysis', ROI_folder, ROI_name, model_version, strcat(ROI_name,'_SnPM_con', num2str(contrast_num )), filesep );

if ~exist(string(data.destination),'dir')
    mkdir(string(data.destination))
end

spm_jobman('initcfg');

if ischar(splitting) && splitting ~= string(model_version)
    folder_files = dir(string(fullfile(path.SPM_folder,'/results', splitting,'ROI_analysis', ROI_folder, ROI_name, model_version, 'sub-*')));
elseif ischar(splitting) && splitting == string(model_version) || ~ischar(splitting)
    folder_files = dir(string(fullfile(path.SPM_folder,'/results', model_version,'ROI_analysis', ROI_folder, ROI_name, 'sub-*')));
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
matlabbatch{1}.spm.tools.snpm.des.OneSampT.ST.ST_later = -1;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.masking.tm.tm_none = 1;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.masking.im = 1;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.masking.em = {strcat(ROI.folder,filesep, ROI.name, ',1')};
matlabbatch{1}.spm.tools.snpm.des.OneSampT.globalc.g_omit = 1;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.globalm.glonorm = 1;

matlabbatch{2}.spm.tools.snpm.cp.snpmcfg(1) = cfg_dep('MultiSub: One Sample T test on diffs/contrasts: SnPMcfg.mat configuration file', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','SnPMcfg'));

spm('defaults', 'fMRI');
spm_jobman('run', matlabbatch);
clear matlabbatch
end
