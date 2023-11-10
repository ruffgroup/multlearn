function GetBetas(SPM_folder, model_version,sub, ROI_folder, ROI)

ROI_name = split(ROI.name,'_');
ROI_name = strcat(ROI_name(1,1), '_',ROI_name(2,1));
ROI_name = split(ROI_name,'.');
ROI_name = ROI_name(1,1);

VOI = fullfile(SPM_folder,'\results', model_version, 'ROIclusters', ROI_folder, [char(ROI_name), '.nii']);

Y = spm_read_vols(spm_vol(VOI),1);
indx = find(Y>0);
[x,y,z] = ind2sub(size(Y),indx);
XYZ = [x y z]';

SPM = load(string(fullfile(SPM_folder,'/results', model_version, sub, 'SPM.mat')));
SPM = SPM.SPM;

if contains(ROI_folder,'choice') || contains(ROI_folder, 'spe')
    Conds = {'speAudio', 'speTactile'};
else
    Conds = {'rpeBestOverallAudio', 'rpeBestOverallTactile'};
end

%Find each occurrence of a trial for a given condition
%These will be stacked together in the Betas array


for cond = 1
    audioBetas = [];
    currCond = Conds{cond};
    if ~iscell(currCond)
        currCond = {currCond};
    end
    for j = 1:length(SPM.Vbeta)
        for k = 1:length(currCond)
            if ~isempty(strfind(SPM.Vbeta(j).descrip,[currCond{k}]))
                Beta = SPM.Vbeta(j).fname;
                P = spm_vol(fullfile(SPM_folder,'/results', model_version, sub,Beta));
                est = spm_get_data(P,XYZ);
                audioBetas = [audioBetas; nanmean(est,2)];
            end
        end
    end


end
for cond = 2
    tactileBetas = [];
    currCond = Conds{cond};
    if ~iscell(currCond)
        currCond = {currCond};
    end
    for j = 1:length(SPM.Vbeta)
        for k = 1:length(currCond)
            if ~isempty(strfind(SPM.Vbeta(j).descrip,[currCond{k}]))
                Beta = SPM.Vbeta(j).fname;
                P = spm_vol(fullfile(SPM_folder,'/results', model_version, sub, Beta));
                est = spm_get_data(P.fname,XYZ);
                tactileBetas = [tactileBetas; nanmean(est,2)];
            end
        end
    end

end

if ~exist(string(fullfile(SPM_folder,'\results', model_version, '\betas', ROI_folder, ROI_name)),'dir')
    mkdir(string(fullfile(SPM_folder,'\results', model_version, '\betas', ROI_folder, ROI_name)));
end
save(string(fullfile(SPM_folder,'\results', model_version, '\betas', ROI_folder, ROI_name, ['audiobetas_' sub '.mat'])),'audioBetas');
save(string(fullfile(SPM_folder,'\results', model_version, '\betas', ROI_folder, ROI_name, ['tactilebetas_' sub '.mat'])),'tactileBetas');
end