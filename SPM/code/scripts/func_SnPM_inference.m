function func_SnPM_inference(path, splitting, contrast_num, model_version, tVal)
tValSplit = split(num2str(tVal),'.');
if ischar(splitting) && splitting ~= string(model_version)
    data.destination = fullfile(path.SPM_folder,'/results', splitting, model_version, strcat('Second_level_SnPM_con', num2str(contrast_num )) );
elseif ischar(splitting) && splitting == string(model_version) || ~ischar(splitting)
    data.destination = fullfile(path.SPM_folder,'/results', model_version,strcat('Second_level_SnPM_con', num2str(contrast_num )) );
end

cd data.destination
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
