% Analyzing functional data
clear all

% SETTINGS:
ISrun_level1 = 0;
ISrun_level2SPM = 1;
ISrun_level2SnPM = 0;
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
models = ["spe", "rpeSimple", "spe_rpeSimple"]; % "other", "spe", "rpeSimple"
tVals = [2.3, 2.6, 3.1];

addpath(fullfile('scripts'));

%% Create conditions file
for i = models % spe, rpeSimple (Init, ...), spe_rpeSimple (same Init, ...)
    model_version = convertStringsToChars(i);
    if ~contains(model_version, "_")
        splitting = false;
        conditions(model_version, splitting)
    else
        splitting = model_version;
        model = split(model_version, "_");
        conditions(char(model(1,1)), splitting)
        conditions(char(model(2,1)), splitting)
        conditions(model_version, splitting)
    end
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

for i = models % spe, rpeSimple (Init, ...), spe_rpeSimple (same Init, ...)
    model_version = convertStringsToChars(i);
    if ISrun_level1 == 1
        parfor (sub = 1:numSubs,M)
        if ~contains(model_version, "_")
            splitting = false;
            func_level1(path.folder_processed, path.bids_folder, path.SPM_folder, subs(sub).name , model_version, splitting)
        else
            splitting = model_version;
            model = split(model_version, "_");
            func_level1(path.folder_processed, path.bids_folder, path.SPM_folder, subs(sub).name , char(model(1,1)), splitting)
            func_level1(path.folder_processed, path.bids_folder, path.SPM_folder, subs(sub).name , char(model(2,1)), splitting)
            func_level1 (path.folder_processed, path.bids_folder, path.SPM_folder, subs(sub).name , model_version, splitting)
        end
        end

    end
% 
     parfor (sub = 1:numSubs,M)
         if model_version == "other"
             func_contrast_level1(path, subs(sub).name, del_old_con);
         elseif ~contains(model_version, "_") && model_version ~= "other"
             splitting = false;
             func_contrast_level1_Pmod(path, subs(sub).name, model_version, del_old_con, splitting);
         else
             splitting = model_version;
             model = split(model_version, "_");
             func_contrast_level1_Pmod(path, subs(sub).name, char(model(1,1)), del_old_con, splitting);
             func_contrast_level1_Pmod(path, subs(sub).name, char(model(2,1)), del_old_con, splitting);
             func_contrast_level1_BothPmods(path, subs(sub).name, model_version, del_old_con);
         end
     end


    if model_version == "other"
        splitting = false;
        num_contrasts = 2; % number set in func_contrast_level1
        
        parfor (k = 1:num_contrasts,M)
            if ISrun_level2SPM == 1
                func_level2(path,subs, model_version, k, splitting);
            end
            if ISrun_level2SnPM == 1
                for tVal = tVals
                func_level2_SnPM(path,subs, model_version, k, splitting, tVal);
                end
            end
        end

    elseif ~contains(model_version, "_") && model_version ~= "other"
        splitting = false;
        num_contrasts = 7;
        parfor (k = 1:num_contrasts,M)
            if ISrun_level2SPM == 1
                func_level2(path,subs, model_version, k, splitting);
            end
            if ISrun_level2SnPM == 1
                for tVal = tVals
                func_level2_SnPM(path,subs, model_version, k, splitting, tVal);
                end
            end
        end
    
    else
        splitting = model_version;
        model = split(model_version, "_");
        num_contrasts1 = 7;
        num_contrasts2 = 7;
        num_contrasts3 = 13;

        parfor (k = 1:num_contrasts1,M)
            if ISrun_level2SPM == 1
                func_level2(path,subs, char(model(1,1)), k, splitting);
            end
            if ISrun_level2SnPM == 1
                for tVal = tVals
                func_level2_SnPM(path,subs, char(model(1,1)), k, splitting);
                end
            end
        end

        parfor (k = 1:num_contrasts2,M)
            if ISrun_level2SPM == 1
                func_level2(path,subs, char(model(2,1)), k, splitting);
            end
            if ISrun_level2SnPM == 1
                for tVal = tVals
                func_level2_SnPM(path,subs, char(model(2,1)), k, splitting, tVal);
                end
            end
        end

        parfor (k = 1:num_contrasts3,M)
            if ISrun_level2SPM == 1
                func_level2(path,subs, model_version, k, splitting);
            end
            if ISrun_level2SnPM == 1
                for tVal = tVals
                func_level2_SnPM(path,subs, model_version, k, splitting, tVal);
                end
            end
        end

    end

    



% 
%     func_level3(path, model_version);

end

%%
