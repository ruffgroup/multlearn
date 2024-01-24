function func_VOI(SPM_folder, model_version, splitting,sub, ROI_folder, ROI, concat)

ROI_name = split(ROI.name,'_');
ROI_name = strcat(ROI_name(1,1), '_',ROI_name(2,1));
ROI_name = split(ROI_name,'.');
ROI_name = ROI_name(1,1);

if contains(ROI_folder,'choice') || contains(ROI_folder, 'feedback')
spm = fullfile(SPM_folder,'\results', model_version,sub, 'SPM.mat');
else
spm = fullfile(SPM_folder,'\results', model_version,splitting,sub, 'SPM.mat');
end

VOI = fullfile(SPM_folder,'\results', model_version, 'ROIclusters', ROI_folder, [char(ROI_name), '.nii']);
vol = spm_vol(VOI);
[Y,XYZ] = spm_read_vols(vol);
dim = size(Y);
Y = reshape(Y,[1 dim(1)*dim(2)*dim(3)]);
[mxv,idx] = max(Y(:));
ind = XYZ(:,idx);
if contains(ROI_folder,'choice')
    contrast = 14;
elseif contains(ROI_folder,'feedback')
    contrast = 15;
else
    contrast = 1;
end

spm_jobman('initcfg');
    for nrun=1:6
matlabbatch{1}.spm.util.voi.spmmat = {spm};
matlabbatch{1}.spm.util.voi.adjust = NaN;
matlabbatch{1}.spm.util.voi.session = nrun;
matlabbatch{1}.spm.util.voi.name = char(ROI_name);
matlabbatch{1}.spm.util.voi.roi{1}.spm.spmmat = {''};
matlabbatch{1}.spm.util.voi.roi{1}.spm.contrast = contrast;
matlabbatch{1}.spm.util.voi.roi{1}.spm.conjunction = 1;
matlabbatch{1}.spm.util.voi.roi{1}.spm.threshdesc = 'none';
matlabbatch{1}.spm.util.voi.roi{1}.spm.thresh = 1;
matlabbatch{1}.spm.util.voi.roi{1}.spm.extent = 0;
matlabbatch{1}.spm.util.voi.roi{1}.spm.mask = struct('contrast', {}, 'thresh', {}, 'mtype', {});
matlabbatch{1}.spm.util.voi.roi{2}.sphere.centre = [ind(1) ind(2) ind(3)];
matlabbatch{1}.spm.util.voi.roi{2}.sphere.radius = 15;
matlabbatch{1}.spm.util.voi.roi{2}.sphere.move.fixed = 1;
matlabbatch{1}.spm.util.voi.roi{3}.sphere.centre = [0 0 0];
matlabbatch{1}.spm.util.voi.roi{3}.sphere.radius = 6;
matlabbatch{1}.spm.util.voi.roi{3}.sphere.move.local.spm = 1;
matlabbatch{1}.spm.util.voi.roi{3}.sphere.move.local.mask = 'i2';
matlabbatch{1}.spm.util.voi.expression = 'i1&i3';
  
        
        spm_jobman('run', matlabbatch);
        clear matlabbatch
    end

end