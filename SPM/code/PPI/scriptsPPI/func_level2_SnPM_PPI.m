function func_level2_SnPM_PPI(path,subs, model_version, contrast_num, splitting, ROI_folder, ROI)
spm_figure('GetWin','Graphics');
% path of data
ROI_name = split(ROI.name,'_');
ROI_name = strcat(ROI_name(1,1), '_',ROI_name(2,1));
ROI_name = split(ROI_name,'.');
ROI_name = ROI_name(1,1);

data.destination = fullfile(path.SPM_folder,'/results',splitting,'PPI', ROI_folder, ROI_name,strcat(ROI_name,'_SnPM_con', num2str(contrast_num )), filesep );


if ~exist(string(data.destination),'dir')
    mkdir(string(data.destination))
end

spm_jobman('initcfg');

folder_files = dir(string(fullfile(path.SPM_folder,'/results', splitting, 'PPI', ROI_folder, ROI_name, 'sub-*')));


valid = [];
for ii = 1:numel(subs)
    if exist(string(fullfile(path.SPM_folder,'/results', splitting, 'PPI', ROI_folder, ROI_name, subs(ii).name, ['con_000' , num2str(contrast_num) ,'.nii'])), 'file')
        id = split(subs(ii).name,'-');
        valid = [valid str2num(char(id(2,1)))];
    end
end

for ii = 1:length(valid)
    contrasts{ii,1} = ([ folder_files(ii).folder, filesep, 'sub-',num2str(valid(ii),'%02d') , filesep, 'con_000' , num2str(contrast_num) ,'.nii,1' ]);
end

matlabbatch{1}.spm.tools.snpm.des.OneSampT.DesignName = 'MultiSub: One Sample T test on diffs/contrasts';
matlabbatch{1}.spm.tools.snpm.des.OneSampT.DesignFile = 'snpm_bch_ui_OneSampT';
matlabbatch{1}.spm.tools.snpm.des.OneSampT.dir = data.destination;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.P = contrasts;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.cov = struct('c', {}, 'cname', {});
matlabbatch{1}.spm.tools.snpm.des.OneSampT.nPerm = 5000;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.vFWHM = [0 0 0];
matlabbatch{1}.spm.tools.snpm.des.OneSampT.bVolm = 1;
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
