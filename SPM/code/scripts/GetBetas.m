function GetBetas(SPM_folder, model_version,sub, ROI_folder, ROI)

ROI_name = split(ROI.name,'_');
ROI_name = strcat(ROI_name(1,1), '_',ROI_name(2,1));
ROI_name = split(ROI_name,'.');
ROI_name = ROI_name(1,1);
data.destination = fullfile(path.SPM_folder,'/results', model_version, 'PPI', ROI_folder, ROI_name, sub);

SPM = load(fullfile(data.destination, 'SPM.mat'));

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
            if ~isempty(strfind(SPM.Vbeta(j).descrip,[currCond{k} '_']))
                audioBetas = char(Betas,SPM.Vbeta(j).fname);
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
            if ~isempty(strfind(SPM.Vbeta(j).descrip,[currCond{k} '_']))
                tactileBetas = char(Betas,SPM.Vbeta(j).fname);
            end
        end
    end


end
save(fullfile(SPM_folder,'\results', model_version, '\betas', ROI_folder, ROI_name, ['beh'],['betas_' sub '.mat']),'audioBetas', 'tactileBetas');
end