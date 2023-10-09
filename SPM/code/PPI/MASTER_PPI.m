% Analyzing functional data
clear all

% SETTINGS:
ISrun_VOI = 0;
ISrun_PPI = 0;
ISrun_level1_PPI = 1;
ISrun_contrasts_PPI = 0;
del_old_con = 0; % delete old contrasts if you re-run only second level analyses
ISrun_level2SnPM_PPI = 0;
ISrun_level2Inference_PPI = 0;

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
%subs = subs(~contains({subs.name},{'sub-08','sub-13','sub-44', 'sub-64'}));
subs = subs(contains({subs.name},{'sub-64'}));
%, ...
%      'sub-09', 'sub-10', 'sub-11', 'sub-12', 
     %'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', ...
%      'sub-19', 'sub-20', 'sub-21', 'sub-22', 'sub-23'
 %'sub-24', 'sub-25', 'sub-26', 'sub-27', ...
%     'sub-28', 'sub-29', 'sub-30', 'sub-31', 'sub-32', 'sub-33'
%subs = subs(contains({subs.name},{'sub-34','sub-43'}));
models = ["spe_rpeBestOverall"]; % "other", "spe", "rpeSimple", "spe_rpeSimple"
tVals = [2.4, 2.6, 3.1];
addpath(fullfile('scriptsPPI'));
numSubs = numel(subs);
M = 4;

for i = models
    model_version = convertStringsToChars(i);
    splitting = model_version;
    model = split(model_version, "_");
    ROI_choice_folder = dir(fullfile(path.SPM_folder,'results',splitting, ...
        'ROIclusters/choice/*.nii'));
    ROI_feedback_folder = dir(fullfile(path.SPM_folder,'results',splitting, ...
        'ROIclusters/feedback/*.nii'));
    ROI_pmod1_folder = dir(fullfile(path.SPM_folder,'results',splitting, ...
        'ROIclusters',char(model(1,1)), '*.nii'));
    ROI_pmod2_folder = dir(fullfile(path.SPM_folder,'results',splitting, ...
        'ROIclusters',char(model(2,1)), '*.nii'));

    if ISrun_VOI == 1

        parfor (sub = 1:numSubs,M)
            func_VOI(path.SPM_folder, model_version, char(model(1,1)),subs(sub).name, 'feedback', ROI_feedback_folder(1), 0)
%             for ROI = ROI_choice_folder'
%                 %try
%                     func_VOI(path.SPM_folder, model_version, char(model(1,1)),subs(sub).name, 'choice', ROI, 0)
%                 %catch
%                 %end
%             end
%             for ROI = ROI_feedback_folder'
%                 %try
%                     func_VOI(path.SPM_folder, model_version, char(model(2,1)),subs(sub).name, 'feedback', ROI, 0)
%                 %catch
%                 %end
%             end
%             for ROI = ROI_pmod1_folder'
%                 %try
%                     func_VOI(path.SPM_folder, model_version, char(model(1,1)),subs(sub).name, char(model(1,1)), ROI, 0)
%                 %catch
%                 %end
%             end
%             for ROI = ROI_pmod2_folder'
%                 %try
%                     func_VOI(path.SPM_folder, model_version, char(model(2,1)),subs(sub).name, char(model(2,1)), ROI, 0)
%                 %catch
%                 %end
%             end
          end

    end

    if ISrun_PPI == 1

        parfor (sub = 1:numSubs,M)
            func_get_PPI(path.SPM_folder, model_version, char(model(2,1)),subs(sub).name, 'feedback', ROI_feedback_folder(1),0)
%              for ROI = ROI_choice_folder'
% %                 %try
%                      func_get_PPI(path.SPM_folder, model_version, char(model(1,1)),subs(sub).name, 'choice', ROI,0)
% %                 %catch
% %                 %end
%             end
%             for ROI = ROI_feedback_folder'
%                 try
%                     func_get_PPI(path.SPM_folder, model_version, char(model(2,1)),subs(sub).name, 'feedback', ROI,0)
%                 catch
%                 end
%             end
%             for ROI = ROI_pmod1_folder'
%                 try
%                     func_get_PPI(path.SPM_folder, model_version, char(model(1,1)),subs(sub).name, char(model(1,1)), ROI,0)
%                 catch
%                 end
%             end
%             for ROI = ROI_pmod2_folder'
%                 try
% 
%                     func_get_PPI(path.SPM_folder, model_version, char(model(2,1)),subs(sub).name, char(model(2,1)), ROI,0)
%                 catch
%                 end
% 
%             end
        end
    end

            if ISrun_level1_PPI == 1
            
            parfor (sub = 1:numSubs,M)
                func_level1_PPI(path.folder_processed, path.SPM_folder, subs(sub).name, model_version, char(model(1,1)), 'feedback', ROI_feedback_folder(1))
%                 for ROI = ROI_choice_folder'
%                     try
%                     func_level1_PPI(path.folder_processed, path.SPM_folder, subs(sub).name, model_version, char(model(1,1)), 'choice', ROI)
%                     catch
%                     end
%                 end
%                 for ROI = ROI_feedback_folder'
%                 try
%                     func_level1_PPI(path.folder_processed, path.SPM_folder, subs(sub).name, model_version, char(model(2,1)), 'feedback', ROI)
%                 catch
%                 end
%                 end
%                 for ROI = ROI_pmod1_folder'
%                     try
% 
%                     func_level1_PPI(path.folder_processed, path.SPM_folder, subs(sub).name, model_version, char(model(1,1)), char(model(1,1)), ROI)
%                     catch
%                     end
%                 end
%                 for ROI = ROI_pmod2_folder'
%                     try
% 
%                     func_level1_PPI(path.folder_processed, path.SPM_folder, subs(sub).name, model_version, char(model(2,1)), char(model(2,1)), ROI)
%                     catch
%                     end
% 
%                 end
            end
            end

       
            if ISrun_contrasts_PPI == 1
            parfor (sub = 1:numSubs,M)
            

                for ROI = ROI_choice_folder'
                
                    func_contrast_level1_PPI(path, subs(sub).name, char(model(1,1)), del_old_con, splitting, 'choice',ROI);
                   
                end
                for ROI = ROI_feedback_folder'
                    
                    func_contrast_level1_PPI(path, subs(sub).name, char(model(2,1)), del_old_con, splitting, 'feedback', ROI);
                end
                for ROI = ROI_pmod1_folder'
          
                    func_contrast_level1_PPI(path, subs(sub).name, char(model(1,1)), del_old_con, splitting, char(model(1,1)),ROI); 
                end
                for ROI = ROI_pmod2_folder'
                    func_contrast_level1_PPI(path, subs(sub).name, char(model(2,1)), del_old_con, splitting, char(model(2,1)), ROI);
                end
            end
            end
            num_contrasts1 = 5;
            num_contrasts2 = 5;

            if ISrun_level2SnPM_PPI == 1
                parfor (k = 1:num_contrasts1,M)
                for ROI = ROI_feedback_folder'
                    func_level2_SnPM_PPI(path,subs, char(model(2,1)), k, splitting,'feedback', ROI);
                end
                for ROI = ROI_choice_folder'
                    func_level2_SnPM_PPI(path,subs, char(model(1,1)), k, splitting,'choice', ROI);
                end
                for ROI = ROI_pmod1_folder'
                    func_level2_SnPM_PPI(path,subs, char(model(1,1)), k, splitting,char(model(1,1)), ROI);
                end
                for ROI = ROI_pmod2_folder'
                    func_level2_SnPM_PPI(path,subs, char(model(2,1)), k, splitting,char(model(2,1)), ROI);
                end
                end

            end

            if ISrun_level2Inference_PPI == 1
                parfor (k = 1:num_contrasts1,M)
                    for ROI = ROI_feedback_folder'
                        for tVal = tVals
                            try
                            func_SnPM_inference_PPI(path, splitting, k, char(model(2,1)), tVal, 'feedback', ROI);
                            catch
                            end
                        end
                    end
                    for ROI = ROI_choice_folder'
                        for tVal = tVals
                            try
                            func_SnPM_inference_PPI(path, splitting, k, char(model(1,1)), tVal, 'choice', ROI);
                            catch
                            end
                        end
                    end
                    for ROI = ROI_pmod1_folder'
                        for tVal = tVals
                            try
                            func_SnPM_inference_PPI(path, splitting, k, char(model(1,1)), tVal, char(model(1,1)), ROI);
                            catch
                            end
                        end
                    end
                    for ROI = ROI_pmod2_folder'
                        for tVal = tVals
                            try
                            func_SnPM_inference_PPI(path, splitting, k, char(model(2,1)), tVal, char(model(2,1)), ROI);
                            catch
                            end
                        end
                    end
                
                end
            end
end



%% 3. run estimation
% level 1 analysis



%%
