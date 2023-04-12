% Analyzing functional data
clear all

% SETTINGS:
ISrun_level1 = 0;
ISrun_level2SPM = 0;
ISrun_level2SnPM = 1;
del_old_con = 1; % delete old contrasts if you re-run only second level analyses

path.folder_processed = dir(fullfile('../../../data/ds-mlearn/derivatives/fmriprep'));
path.folder_processed = [path.folder_processed(1).folder '/'];
path.bids_folder = dir(fullfile('../../../data/ds-mlearn'));
path.bids_folder = [path.bids_folder(1).folder '/'];
path.SPM_folder = dir(fullfile('../../SPM'));
path.SPM_folder = [path.SPM_folder(1).folder '/'];


subs = dir(fullfile(path.folder_processed, 'sub-*'));
subs = subs([subs.isdir]');
subs = subs(~contains({subs.name},{'sub-08', 'sub-13', 'sub-44'}));

addpath(fullfile('scripts'));

%% Create conditions file
for i = ["SPE"] % SPE, RPE, ALL
    model_version = i;
    conditions(i)
end

%% 1. run tapas

% parfor ii=1:numel(subs)
%     for nrun = 1:6 %number of runs
%         [ii  nrun]
%         func_tapas(subs(ii).name, nrun, path.folder_processed, path.bids_folder);
% 
%     end
% end

%%
% 2. Smooth
% 
% TotalNumRuns = 6;
% parfor (ii=1:numel(subs),4)
%     ii
%     func_smooth(subs(ii).folder, subs(ii).name, TotalNumRuns);
% end

%% 3. run estimation
% level 1 analysis
M = 4; %use 4 cores


numSubs = numel(subs);

for i = ["SPE"] % SPE, RPE, ALL
    model_version = i;
    if ISrun_level1 == 1
        parfor (sub = 1:numSubs,M)
            func_level1(path.folder_processed, path.bids_folder, path.SPM_folder, subs(sub).name , model_version)
        end

    end
% 
     parfor (sub = 1:numSubs,M)
         func_contrast_level1(path, subs(sub).name, model_version, del_old_con);
     end



    num_contasts = 7; % number set in func_contrast_level1

    parfor (k = 1:num_contasts,M)
        if ISrun_level2SPM == 1
            func_level2(path,subs, model_version, k);
        end
        if ISrun_level2SnPM == 1
            func_SnPM(path,subs, model_version, k);
        end
    end



% 
%     func_level3(path, model_version);

end

%%













