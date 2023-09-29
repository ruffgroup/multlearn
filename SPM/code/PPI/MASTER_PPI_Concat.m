% Analyzing functional data
clear all

% SETTINGS:
ISrun_level1_ROI = 1;
ISrun_contrasts_ROI = 0;
del_old_con = 0; % delete old contrasts if you re-run only second level analyses
ISrun_level2_ROI = 0;
ISrun_level2Inference_ROI = 0;

path.folder_processed = dir(fullfile('../../../../data/ds-mlearn/derivatives/fmriprep'));
path.folder_processed = [path.folder_processed(1).folder '/'];
path.bids_folder = dir(fullfile('../../../../data/ds-mlearn'));
path.bids_folder = [path.bids_folder(1).folder '/'];
path.SPM_folder = dir(fullfile('../../../SPM'));
path.SPM_folder = [path.SPM_folder(1).folder '/'];
path.neurosynth_folder = dir(fullfile(path.SPM_folder,'neurosynth/*.img'));
%path.neurosynth_folder = [path.neurosynth_folder(1).folder '/'];


subs = dir(fullfile(path.folder_processed, 'sub-*'));
subs = subs([subs.isdir]');
subs = subs(~contains({subs.name},{'sub-08', 'sub-13', 'sub-44'}));
models = ["spe_rpeBestOverall"]; % "other", "spe", "rpeSimple", "spe_rpeSimple"
ROI_tVals = [1];
addpath(fullfile('scriptsPPI'));

% Create conditions file
% for i = models % spe, rpeSimple (Init, ...), spe_rpeSimple (same Init, ...)
%     model_version = convertStringsToChars(i);
%     func_setup_PPI(model_version)
% end

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

        splitting = model_version;
        model = split(model_version, "_");
        num_contrasts1 = 7;
        num_contrasts2 = 7;
        num_contrasts3 = 15;
%         ROI_audio_folder = dir(fullfile(path.SPM_folder,'results',splitting, ...
%                     'ROIclusters/audio/*.img'));
%         ROI_tactile_folder = dir(fullfile(path.SPM_folder,'results',splitting, ...
%                     'ROIclusters/tactile/*.img'));
%         ROI_choice_folder = dir(fullfile(path.SPM_folder,'results',splitting, ...
%                     'ROIclusters/choice/*.img'));
%         ROI_feedback_folder = dir(fullfile(path.SPM_folder,'results',splitting, ...
%                     'ROIclusters/feedback/*.img'));
        ROI_pmod1_folder = dir(fullfile(path.SPM_folder,'results',splitting, ...
                    'ROIclusters',char(model(1,1)), '*.img'));
        ROI_pmod2_folder = dir(fullfile(path.SPM_folder,'results',splitting, ...
                    'ROIclusters',char(model(2,1)), '*.img'));

        if ISrun_level1_ROI == 1
            parfor (sub = 1:numSubs,M)
%                 for ROI = path.neurosynth_folder'
%                     try
%                     func_level1_ROI(path.folder_processed, path.bids_folder, path.SPM_folder, subs(sub).name, model_version, 'neurosynth', ROI)
%                     catch
%                     end
%                 end
%                 for ROI = ROI_audio_folder'
%                     try
%                     func_level1_ROI(path.folder_processed, path.bids_folder, path.SPM_folder, subs(sub).name, model_version, 'audio', ROI)
%                     catch
%                     end
%                 end
%                 for ROI = ROI_tactile_folder'
%                     try
% 
%                     func_level1_ROI(path.folder_processed, path.bids_folder, path.SPM_folder, subs(sub).name, model_version, 'tactile', ROI)
%                     catch
%                     end
%                 end
%                 for ROI = ROI_choice_folder'
%                     try
%                     func_level1_ROI(path.folder_processed, path.bids_folder, path.SPM_folder, subs(sub).name, model_version, 'choice', ROI)
%                     catch
%                     end
%                 end
%                 for ROI = ROI_feedback_folder'
%                 try
%                     func_level1_ROI(path.folder_processed, path.bids_folder, path.SPM_folder, subs(sub).name, model_version, 'feedback', ROI)
%                 catch
%                 end
%                 end
                for ROI = ROI_pmod1_folder'
                    try

                    func_level1_PPI(path.folder_processed, path.bids_folder, path.SPM_folder, subs(sub).name, model_version, char(model(1,1)), ROI)
                    catch
                    end
                end
                for ROI = ROI_pmod2_folder'
                    try

                    func_level1_PPI(path.folder_processed, path.bids_folder, path.SPM_folder, subs(sub).name, model_version, char(model(2,1)), ROI)
                    catch
                    end

                end
            end
        end


        if ISrun_contrasts_ROI == 1
            parfor (sub = 1:numSubs,M)
%                 for ROI = path.neurosynth_folder'
%                     func_contrast_level1_Pmod_ROI(path, subs(sub).name, char(model(1,1)), del_old_con, splitting, 'neurosynth',ROI);
%                     func_contrast_level1_Pmod_ROI(path, subs(sub).name, char(model(2,1)), del_old_con, splitting, 'neurosynth', ROI);
%                 end
%                 for ROI = ROI_audio_folder'
%                     func_contrast_level1_Pmod_ROI(path, subs(sub).name, char(model(1,1)), del_old_con, splitting, 'audio',ROI);
%                     func_contrast_level1_Pmod_ROI(path, subs(sub).name, char(model(2,1)), del_old_con, splitting, 'audio', ROI);
%                 end
%                 for ROI = ROI_tactile_folder'
%                     func_contrast_level1_Pmod_ROI(path, subs(sub).name, char(model(1,1)), del_old_con, splitting, 'tactile',ROI);
%                     func_contrast_level1_Pmod_ROI(path, subs(sub).name, char(model(2,1)), del_old_con, splitting, 'tactile', ROI);
%                 end
%                 for ROI = ROI_choice_folder'
%                     func_contrast_level1_Pmod_ROI(path, subs(sub).name, char(model(1,1)), del_old_con, splitting, 'choice',ROI);
%                     %func_contrast_level1_Pmod_ROI(path, subs(sub).name, char(model(2,1)), del_old_con, splitting, 'choice', ROI);
%                 end
%                 for ROI = ROI_feedback_folder'
%                     %func_contrast_level1_Pmod_ROI(path, subs(sub).name, char(model(1,1)), del_old_con, splitting, 'feedback',ROI);
%                     func_contrast_level1_Pmod_ROI(path, subs(sub).name, char(model(2,1)), del_old_con, splitting, 'feedback', ROI);
%                 end
                for ROI = ROI_pmod1_folder'
                    func_contrast_level1_Pmod_ROI(path, subs(sub).name, char(model(1,1)), del_old_con, splitting, char(model(1,1)),ROI);
                    %func_contrast_level1_Pmod_ROI(path, subs(sub).name, char(model(2,1)), del_old_con, splitting, char(model(1,1)), ROI);
                end
                for ROI = ROI_pmod2_folder'
                    %func_contrast_level1_Pmod_ROI(path, subs(sub).name, char(model(1,1)), del_old_con, splitting, char(model(2,1)),ROI);
                    func_contrast_level1_Pmod_ROI(path, subs(sub).name, char(model(2,1)), del_old_con, splitting, char(model(2,1)), ROI);
                end
            end
        end


        parfor (k = 1:num_contrasts1,M)
            if ISrun_level2SnPM == 1
                func_level2_SnPM(path,subs, char(model(1,1)), k, splitting);
            end
            if ISrun_level2Inference == 1
                for tVal = tVals
                    try
                        func_SnPM_inference(path,splitting, k,char(model(1,1)),tVal)
                    catch
                    end
                end
            end
            if ISrun_level2_ROI == 1
                ROI_Pmod_folder = dir(fullfile(path.SPM_folder,'results',splitting, ...
                    'ROIclusters',char(model(1,1)), '*.img'));
                if k == 4 || k == 3 || k == 6
                    for ROI = path.neurosynth_folder'
                        func_level2_SnPM_ROI_Pmod(path,subs, char(model(1,1)), k, splitting, 'neurosynth',ROI);
                        if ISrun_level2Inference_ROI == 1
                            for tVal = ROI_tVals
                                try
                                    func_SnPM_inference_ROI(path, splitting, k, char(model(1,1)), tVal, 'neurosynth', ROI);
                                catch
                                end
                            end
                        end
                    end
                    for ROI = ROI_Pmod_folder'
                        func_level2_SnPM_ROI_Pmod(path,subs, char(model(1,1)), k, splitting,char(model(1,1)), ROI);
                        if ISrun_level2Inference_ROI == 1
                            for tVal = ROI_tVals
                                try
                                    func_SnPM_inference_ROI(path, splitting, k, char(model(1,1)), tVal, char(model(1,1)), ROI);
                                catch
                                end
                            end
                        end
                    end
                    for ROI = ROI_tactile_folder'
                        func_level2_SnPM_ROI_Pmod(path,subs, char(model(1,1)), k, splitting,'tactile', ROI);
                        if ISrun_level2Inference_ROI == 1
                            for tVal = ROI_tVals
                                try
                                    func_SnPM_inference_ROI(path, splitting, k, char(model(1,1)), tVal, 'tactile', ROI);
                                catch
                                end
                            end
                        end
                    end
                    for ROI = ROI_choice_folder'
                        func_level2_SnPM_ROI_Pmod(path,subs, char(model(1,1)), k, splitting,'choice', ROI);
                        if ISrun_level2Inference_ROI == 1
                            for tVal = ROI_tVals
                                try
                                    func_SnPM_inference_ROI(path, splitting, k, char(model(1,1)), tVal, 'choice', ROI);
                                catch
                                end
                            end
                        end
                    end
                elseif k == 5 || k == 2 || k == 7
                    for ROI = path.neurosynth_folder'
                        func_level2_SnPM_ROI_Pmod(path,subs, char(model(1,1)), k, splitting, 'neurosynth', ROI);
                        if ISrun_level2Inference_ROI == 1
                            for tVal = ROI_tVals
                                try
                                    func_SnPM_inference_ROI(path, splitting, k, char(model(1,1)), tVal, 'neurosynth', ROI);
                                catch
                                end
                            end
                        end
                    end
                    for ROI = ROI_Pmod_folder'
                        func_level2_SnPM_ROI_Pmod(path,subs, char(model(1,1)), k, splitting,char(model(1,1)), ROI);
                        if ISrun_level2Inference_ROI == 1
                            for tVal = ROI_tVals
                                try
                                    func_SnPM_inference_ROI(path, splitting, k, char(model(1,1)), tVal, char(model(1,1)), ROI);
                                catch
                                end
                            end
                        end
                    end
                    for ROI = ROI_audio_folder'
                        func_level2_SnPM_ROI_Pmod(path,subs, char(model(1,1)), k, splitting,'audio', ROI);
                        if ISrun_level2Inference_ROI == 1
                            for tVal = ROI_tVals
                                try
                                    func_SnPM_inference_ROI(path, splitting, k, char(model(1,1)), tVal, 'audio', ROI);
                                catch
                                end
                            end
                        end
                    end
                    for ROI = ROI_choice_folder'
                        func_level2_SnPM_ROI_Pmod(path,subs, char(model(1,1)), k, splitting,'choice', ROI);
                        if ISrun_level2Inference_ROI == 1
                            for tVal = ROI_tVals
                                try
                                    func_SnPM_inference_ROI(path, splitting, k, char(model(1,1)), tVal, 'choice', ROI);
                                catch
                                end
                            end
                        end
                    end
                end
            end
        end

        parfor (k = 1:num_contrasts2,M)
            if ISrun_level2SPM == 1
                func_level2(path,subs, char(model(2,1)), k, splitting);
            end
            if ISrun_level2SnPM == 1
                func_level2_SnPM(path,subs, char(model(2,1)), k, splitting);
            end
            if ISrun_level2Inference == 1
                for tVal = tVals
                    try
                        func_SnPM_inference(path,splitting, k,char(model(2,1)),tVal)
                    catch
                    end
                end
            end
            
            if ISrun_level2_ROI == 1
                ROI_Pmod_folder = dir(fullfile(path.SPM_folder,'results',splitting, ...
                    'ROIclusters',char(model(2,1)), '*.img'));
                if k == 4 || k == 3 || k == 6
                    for ROI = path.neurosynth_folder'
                        func_level2_SnPM_ROI_Pmod(path,subs, char(model(2,1)), k, splitting, 'neurosynth',ROI);
                        if ISrun_level2Inference_ROI == 1
                            for tVal = ROI_tVals
                                try
                                    func_SnPM_inference_ROI(path, splitting, k, char(model(2,1)), tVal, 'neurosynth', ROI);
                                catch
                                end
                            end
                        end
                    end
                    for ROI = ROI_Pmod_folder'
                        func_level2_SnPM_ROI_Pmod(path,subs, char(model(2,1)), k, splitting,char(model(2,1)), ROI);
                        if ISrun_level2Inference_ROI == 1
                            for tVal = ROI_tVals
                                try
                                    func_SnPM_inference_ROI(path, splitting, k, char(model(2,1)), tVal, char(model(2,1)), ROI);
                                catch
                                end
                            end
                        end
                    end
                    for ROI = ROI_tactile_folder'
                        func_level2_SnPM_ROI_Pmod(path,subs, char(model(2,1)), k, splitting,'tactile', ROI);
                        if ISrun_level2Inference_ROI == 1
                            for tVal = ROI_tVals
                                try
                                    func_SnPM_inference_ROI(path, splitting, k, char(model(2,1)), tVal, 'tactile', ROI);
                                catch
                                end
                            end
                        end
                    end
                    for ROI = ROI_feedback_folder'
                        func_level2_SnPM_ROI_Pmod(path,subs, char(model(2,1)), k, splitting,'feedback', ROI);
                        if ISrun_level2Inference_ROI == 1
                            for tVal = ROI_tVals
                                try
                                    func_SnPM_inference_ROI(path, splitting, k, char(model(2,1)), tVal, 'feedback', ROI);
                                catch
                                end
                            end
                        end
                    end
                elseif k == 5 || k == 2 || k == 7
                    for ROI = path.neurosynth_folder'
                        func_level2_SnPM_ROI_Pmod(path,subs, char(model(2,1)), k, splitting, 'neurosynth', ROI);
                        if ISrun_level2Inference_ROI == 1
                            for tVal = ROI_tVals
                                try
                                    func_SnPM_inference_ROI(path, splitting, k, char(model(2,1)), tVal, 'neurosynth', ROI);
                                catch
                                end
                            end
                        end
                    end
                    for ROI = ROI_Pmod_folder'
                        func_level2_SnPM_ROI_Pmod(path,subs, char(model(2,1)), k, splitting,char(model(2,1)), ROI);
                        if ISrun_level2Inference_ROI == 1
                            for tVal = ROI_tVals
                                try
                                    func_SnPM_inference_ROI(path, splitting, k, char(model(2,1)), tVal, char(model(2,1)), ROI);
                                catch
                                end
                            end
                        end
                    end
                    for ROI = ROI_audio_folder'
                        func_level2_SnPM_ROI_Pmod(path,subs, char(model(2,1)), k, splitting,'audio', ROI);
                        if ISrun_level2Inference_ROI == 1
                            for tVal = ROI_tVals
                                try
                                    func_SnPM_inference_ROI(path, splitting, k, char(model(2,1)), tVal, 'audio', ROI);
                                catch
                                end
                            end
                        end
                    end
                    for ROI = ROI_feedback_folder'
                        func_level2_SnPM_ROI_Pmod(path,subs, char(model(2,1)), k, splitting,'feedback', ROI);
                        if ISrun_level2Inference_ROI == 1
                            for tVal = ROI_tVals
                                try
                                    func_SnPM_inference_ROI(path, splitting, k, char(model(2,1)), tVal, 'feedback', ROI);
                                catch
                                end
                            end
                        end
                    end
                end
            end
        end

        parfor (k = 1:num_contrasts3,M)
            if ISrun_level2SPM == 1
                func_level2(path,subs, model_version, k, splitting);
            end
            if ISrun_level2SnPM == 1
                func_level2_SnPM(path,subs, model_version, k, splitting);
            end
            if ISrun_level2Inference == 1
                for tVal = tVals
                    try
                        func_SnPM_inference(path,splitting, k,model_version,tVal)
                    catch
                    end
                end
            end
            
%             if ISrun_level2_ROI == 1
%                 func_level2_SnPM_ROI(path,subs, model_version, k, splitting, ROI);
%             end
        end



% 
%     func_level3(path, model_version);

end

%%
