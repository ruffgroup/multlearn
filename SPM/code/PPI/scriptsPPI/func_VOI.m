function func_VOI(SPM_folder, model_version, splitting,sub, ROI_folder, ROI, concat)

ROI_name = split(ROI.name,'_');
ROI_name = strcat(ROI_name(1,1), '_',ROI_name(2,1));
VOIspm = fullfile(SPM_folder,'\results', model_version, 'ROI_analysis', ROI_folder, char(ROI_name), splitting, sub, 'SPM.mat');
VOIdir = dir(fullfile(SPM_folder,'\results', model_version, 'ROIclusters', ROI_folder, [char(ROI_name),'*.img']));
VOI = spm_vol(fullfile(VOIdir(1).folder, VOIdir(1).name));
TH = 0;
[Y,XYZ] = spm_read_vols(VOI);
idx=find(Y(:)>TH);
XYZmm=XYZ(:,idx);
center=mean(XYZmm,2);

spm_jobman('initcfg');
if concat == 0
    for nrun=1:6
        matlabbatch{1}.spm.util.voi.spmmat = cellstr(VOIspm);
        matlabbatch{1}.spm.util.voi.adjust = NaN;
        matlabbatch{1}.spm.util.voi.session = nrun;
        matlabbatch{1}.spm.util.voi.name = char(ROI_name);
        matlabbatch{1}.spm.util.voi.roi{1}.spm.spmmat = {''};
        matlabbatch{1}.spm.util.voi.roi{1}.spm.contrast = 1;
        matlabbatch{1}.spm.util.voi.roi{1}.spm.conjunction = 1;
        matlabbatch{1}.spm.util.voi.roi{1}.spm.threshdesc = 'none';
        matlabbatch{1}.spm.util.voi.roi{1}.spm.thresh = 1;
        matlabbatch{1}.spm.util.voi.roi{1}.spm.extent = 0;
        matlabbatch{1}.spm.util.voi.roi{1}.spm.mask = struct('contrast', {}, 'thresh', {}, 'mtype', {});
        matlabbatch{1}.spm.util.voi.roi{2}.sphere.centre = center; % Set coordinates here
        matlabbatch{1}.spm.util.voi.roi{2}.sphere.radius = 30;           % Radius (mm)
        matlabbatch{1}.spm.util.voi.roi{2}.sphere.move.fixed = 1;
        matlabbatch{1}.spm.util.voi.roi{3}.sphere.centre = [0 0 0];
        matlabbatch{1}.spm.util.voi.roi{3}.sphere.radius = 6;
        matlabbatch{1}.spm.util.voi.roi{3}.sphere.move.local.spm = 1;
        matlabbatch{1}.spm.util.voi.roi{3}.sphere.move.local.mask = 'i2';
        matlabbatch{1}.spm.util.voi.expression = 'i1 & i3';

        spm('defaults', 'fMRI');
        spm_jobman('run', matlabbatch);
        clear matlabbatch
    end
else
    matlabbatch{1}.spm.util.voi.spmmat = cellstr(VOIspm);
    matlabbatch{1}.spm.util.voi.adjust = NaN;
    matlabbatch{1}.spm.util.voi.session = 1;
    matlabbatch{1}.spm.util.voi.name = char(ROI_name);
    matlabbatch{1}.spm.util.voi.roi{1}.spm.spmmat = {''};
    matlabbatch{1}.spm.util.voi.roi{1}.spm.contrast = 1;
    matlabbatch{1}.spm.util.voi.roi{1}.spm.conjunction = 1;
    matlabbatch{1}.spm.util.voi.roi{1}.spm.threshdesc = 'none';
    matlabbatch{1}.spm.util.voi.roi{1}.spm.thresh = 1;
    matlabbatch{1}.spm.util.voi.roi{1}.spm.extent = 0;
    matlabbatch{1}.spm.util.voi.roi{1}.spm.mask = struct('contrast', {}, 'thresh', {}, 'mtype', {});
    matlabbatch{1}.spm.util.voi.roi{2}.sphere.centre = [0 0 0];
    matlabbatch{1}.spm.util.voi.roi{2}.sphere.radius = 6;
    matlabbatch{1}.spm.util.voi.roi{2}.sphere.move.global.spm = 1;
    matlabbatch{1}.spm.util.voi.roi{2}.sphere.move.global.mask = '';
    matlabbatch{1}.spm.util.voi.expression = 'i1 & i2';

    spm('defaults', 'fMRI');
    spm_jobman('run', matlabbatch);
    clear matlabbatch
end
end