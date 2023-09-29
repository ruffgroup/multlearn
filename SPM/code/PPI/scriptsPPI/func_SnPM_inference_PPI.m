function func_SnPM_inference_PPI(path, splitting, contrast_num, model_version, tVal, ROI_folder, ROI)
tValSplit = split(num2str(tVal),'.');
ROI_name = split(ROI.name,'_');
ROI_name = strcat(ROI_name(1,1), '_',ROI_name(2,1));
data.destination = fullfile(path.SPM_folder,'/results',splitting,'PPI', ROI_folder, ROI_name,strcat(ROI_name,'_SnPM_con', num2str(contrast_num )), filesep );

spm_jobman('initcfg');
matlabbatch{1}.spm.tools.snpm.inference.SnPMmat = cellstr(string(fullfile(data.destination, filesep, 'SnPM.mat')));
matlabbatch{1}.spm.tools.snpm.inference.Thr.Clus.ClusSize.CFth = tVal;
matlabbatch{1}.spm.tools.snpm.inference.Thr.Clus.ClusSize.ClusSig.FWEthC = 0.05;
matlabbatch{1}.spm.tools.snpm.inference.Tsign = 1;
matlabbatch{1}.spm.tools.snpm.inference.WriteFiltImg.name = char(strcat(string(data.destination),filesep,'SnPM_filtered_', tValSplit(1,1),tValSplit(2,1)));
matlabbatch{1}.spm.tools.snpm.inference.Report = 'MIPtable';

spm('defaults', 'fMRI');
spm_jobman('run', matlabbatch);
savefig(char(strcat(string(data.destination),filesep,'SnPM_filtered_',tValSplit(1,1),tValSplit(2,1),'.fig')))
clear matlabbatch
