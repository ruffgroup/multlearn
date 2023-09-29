function conditions_PPI(model_version)

%subjects = [01 02 03 04 05 06 07 09 10 11 12 14 15 17 18 19 20 21 22 23 24 25 26 27 28 29 30 33 34 35 36 37 38 39 40 41 42 43 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64];
subjects = [01 02 03 04 05 06];
for subject=subjects
    
subject = num2str(subject, '%02d');
cd(['/data/ds-mlearn/derivatives/fmriprep/sub-' subject '/func']) % Navigate to the subject's directory

for run=1:6

events = tdfread(['sub-' subject '_task-learn_run-' num2str(run) '_events.tsv'], '\t'); % Read onset times file
events.trial_type = strtrim(string(events.trial_type)); % Convert char array to string array, to make logical comparisons easier
events.runType = strtrim(string(events.runType));
runType = events.runType(1,:);

choiceIdx = find(events.trial_type == "choice");
feedbackIdx = find(events.trial_type == "feedback");

if contains(model_version, "spe") && ~contains(model_version, "rpe") 
    spe = load(['/data/fittedParameters/sub-' subject '/' model_version '.mat']).spe;

elseif contains(model_version, "rpe") && ~contains(model_version, "spe")
    if contains(model_version, "Best")
        rpe = load(['/data/fittedParameters/sub-' subject '/' model_version '.mat']).rpe.rpe;
    else
        rpe = load(['/data/fittedParameters/sub-' subject '/' model_version '.mat']).rpe;
    end

elseif contains(model_version, "_")
    model = split(model_version, '_');
    spe = load(['/data/fittedParameters/sub-' subject '/' char(model(1,1)) '.mat']).spe;
    if contains(model_version, "Best")
        rpe = load(['/data/fittedParameters/sub-' subject '/' char(model(2,1)) '.mat']).rpe.rpe;
    else
        rpe = load(['/data/fittedParameters/sub-' subject '/' char(model(2,1)) '.mat']).rpe;
    end
end


if runType == "tactile"
    names{1} = 'ChoiceTactile';
    names{2} = 'FeedbackTactile';
    if contains(model_version, "spe") && ~contains(model_version, "rpe")
        pmod = struct('name', {}, 'param', {}, 'poly', {});
        pmod(1).name{1}  = strcat(model_version, 'Tactile');
        pmod(1).param{1} = fillmissing(spe(run,:) - nanmean(spe(run,:)),"constant",0);
        pmod(1).poly{1}  = 1;
    elseif contains(model_version, "rpe") && ~contains(model_version, "spe")
        pmod = struct('name', {}, 'param', {}, 'poly', {});
        pmod(2).name{1} = strcat(model_version, 'Tactile');
        pmod(2).param{1} = fillmissing(rpe(run,:) - nanmean(rpe(run,:)),"constant",0);
        pmod(2).poly{1}  = 1;
    elseif contains(model_version, "_")
        pmod = struct('name', {}, 'param', {}, 'poly', {});
        model = split(model_version, '_');
        pmod(1).name{1}  = strcat(string(model(1,1)), 'Tactile');
        pmod(1).param{1} = fillmissing(spe(run,:) - nanmean(spe(run,:)),"constant",0);
        pmod(1).poly{1}  = 1;

        pmod(2).name{1} = strcat(string(model(2,1)), 'Tactile');
        pmod(2).param{1} = fillmissing(rpe(run,:) - nanmean(rpe(run,:)),"constant",0);
        pmod(2).poly{1}  = 1;
    end
elseif runType == "audio"
    names{1} = 'ChoiceAudio';
    names{2} = 'FeedbackAudio';
    if contains(model_version, "spe") && ~contains(model_version, "rpe")
         pmod = struct('name', {}, 'param', {}, 'poly', {});
        pmod(1).name{1}  = strcat(model_version, 'Audio');
        pmod(1).param{1} = fillmissing(spe(run,:) - nanmean(spe(run,:)),"constant",0);
        pmod(1).poly{1}  = 1;
    elseif contains(model_version, "rpe") && ~contains(model_version, "spe") 
         pmod = struct('name', {}, 'param', {}, 'poly', {});
        pmod(2).name{1} = strcat(model_version, 'Audio');
        pmod(2).param{1} = fillmissing(rpe(run,:) - nanmean(rpe(run,:)),"constant",0);
        pmod(2).poly{1}  = 1;
    elseif contains(model_version, "_")
         pmod = struct('name', {}, 'param', {}, 'poly', {});
        model = split(model_version, '_');
        pmod(1).name{1}  = strcat(string(model(1,1)), 'Audio');
        pmod(1).param{1} = fillmissing(spe(run,:) - nanmean(spe(run,:)),"constant",0);
        pmod(1).poly{1}  = 1;

        pmod(2).name{1} = strcat(string(model(2,1)), 'Audio');
        pmod(2).param{1} = fillmissing(rpe(run,:) - nanmean(rpe(run,:)),"constant",0);
        pmod(2).poly{1}  = 1;
    end
end
names{3} = 'VOI';

onsets{1} = events.onset(choiceIdx);
durations{1} = events.duration(choiceIdx);
onsets{2} = events.onset(feedbackIdx);
durations{2} = events.duration(feedbackIdx);

orth{1} = false;
orth{2} = false;



destination = fullfile(['/data/ds-mlearn/derivatives/fmriprep/sub-' num2str(subject)], ['beh'], [model_version]);

if ~exist(destination,'dir')
    mkdir(destination)
end
if model_version == "other"
    save(fullfile(destination, ['/run_' num2str(run) '_conditions.mat']), 'names', 'onsets', 'durations')
else
    save(fullfile(destination, ['/run_' num2str(run) '_conditions.mat']), 'names', 'onsets', 'durations', 'pmod', 'orth')
end

end
end
end