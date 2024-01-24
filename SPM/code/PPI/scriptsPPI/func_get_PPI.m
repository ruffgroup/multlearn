function func_get_PPI(SPM_folder, model_version, splitting,sub, ROI_folder, ROI, concat)

ROI_name = split(ROI.name,'_');
ROI_name = strcat(ROI_name(1,1), '_',ROI_name(2,1));
ROI_name = split(ROI_name,'.');
ROI_name = ROI_name(1,1);

if contains(ROI_folder,'choice') || contains(ROI_folder, 'feedback')
spmfolder = string(fullfile(SPM_folder, '/results', model_version,sub));
else
spmfolder = string(fullfile(SPM_folder, '/results', model_version,splitting, sub));
end
VOIfolder = string(fullfile(SPM_folder, '/results', model_version,sub));


spm('defaults', 'fMRI');
spm_jobman('initcfg');
%if concat == 0
    for nrun=1:6
        if isfile(fullfile(VOIfolder, ['VOI_',char(ROI_name),'_', num2str(nrun),'.mat']))
        matlabbatch{1}.spm.stats.ppi.spmmat = cellstr(fullfile(spmfolder, 'SPM.mat'));
        matlabbatch{1}.spm.stats.ppi.type.ppi.voi = cellstr(fullfile(VOIfolder, ['VOI_',char(ROI_name),'_', num2str(nrun),'.mat']));
        if contains(splitting,'spe')
            matlabbatch{1}.spm.stats.ppi.type.ppi.u = [1 1 0
                1 2 1
                2 1 0
                2 2 0];
            matlabbatch{1}.spm.stats.ppi.name = strcat(char(ROI_name),'_',num2str(nrun));
        elseif contains(splitting,'rpe')
            matlabbatch{1}.spm.stats.ppi.type.ppi.u = [1 1 0
                1 2 0
                2 1 0
                2 2 1];
            matlabbatch{1}.spm.stats.ppi.name = strcat(char(ROI_name),'_',num2str(nrun));
        end
        matlabbatch{1}.spm.stats.ppi.disp = 0;
        spm_jobman('run', matlabbatch);
        clear matlabbatch
        end
    end
%end