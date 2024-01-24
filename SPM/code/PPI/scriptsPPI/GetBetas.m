function GetBetas(SPM_folder, model_version,sub, ROI_folder, ROI)

ROI_name = split(ROI.name,'_');
ROI_name = strcat(ROI_name(1,1), '_',ROI_name(2,1));
ROI_name = split(ROI_name,'.');
ROI_name = ROI_name(1,1);

if contains(ROI_folder,'choice') || contains(ROI_folder, 'spe')
    Conds = {'speAudio', 'speTactile'};
    con_folder = 'spe';
else
    Conds = {'rpeBestOverallAudio', 'rpeBestOverallTactile'};
    con_folder = 'rpeBestOverall';
end

SPM = load(string(fullfile(SPM_folder,'/results', model_version, sub, 'SPM.mat')));
SPM = SPM.SPM;
% Betas = [];
% for j = 1:length(SPM.Vbeta)
%     for k = 1:length(Conds)
%         if ~isempty(strfind(SPM.Vbeta(j).descrip,[Conds{k}]))
%             Beta = SPM.Vbeta(j).fname;
%             Betas = [Betas; Beta];
%         end
%     end
% end



%audioBetas = [];
audio2Betas = [];
%tactileBetas = [];
tactile2Betas = [];
for nrun=1:6
    Y = spm_read_vols(spm_vol(fullfile(SPM_folder,'\results', model_version, sub, ['VOI_',char(ROI_name),'_', num2str(nrun),'_eigen.nii'])),1);
    indx = find(Y>0);
    [x,y,z] = ind2sub(size(Y),indx);
    XYZ = [x y z]';

    %P = spm_vol(fullfile(SPM_folder,'/results', model_version, sub,Betas(nrun,:))).fname;
    %est = spm_get_data(P,XYZ);
    if numel(find(contains(SPM.xX.name,['Sn(', num2str(nrun), ') ', 'ChoiceAudio']))) > 0
        %audioBetas = [audioBetas; nanmean(est,2)];
        P2 = spm_vol(fullfile(SPM_folder,'/results', model_version,con_folder, sub,'con_0002.nii'));
        est2 = spm_get_data(P2,XYZ);
        audio2Betas = [audio2Betas; nanmean(est2,2)];
    else
        %tactileBetas = [tactileBetas; nanmean(est,2)];
        P2 = spm_vol(fullfile(SPM_folder,'/results', model_version,con_folder, sub,'con_0003.nii'));
        est2 = spm_get_data(P2,XYZ);
        tactile2Betas = [tactile2Betas; nanmean(est2,2)];
    end
end

if ~exist(string(fullfile(SPM_folder,'\results', model_version, '\betas', ROI_folder, ROI_name)),'dir')
    mkdir(string(fullfile(SPM_folder,'\results', model_version, '\betas', ROI_folder, ROI_name)));
end
%save(string(fullfile(SPM_folder,'\results', model_version, '\betas', ROI_folder, ROI_name, ['audiobetas_' sub '.mat'])),'audioBetas');
save(string(fullfile(SPM_folder,'\results', model_version, '\betas', ROI_folder, ROI_name, ['audio2betas_' sub '.mat'])),'audio2Betas');
%save(string(fullfile(SPM_folder,'\results', model_version, '\betas', ROI_folder, ROI_name, ['tactilebetas_' sub '.mat'])),'tactileBetas');
save(string(fullfile(SPM_folder,'\results', model_version, '\betas', ROI_folder, ROI_name, ['tactile2betas_' sub '.mat'])),'tactile2Betas');
end