<<<<<<< HEAD
function conditions(model_version)

subjects = [01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 17 18 19 20 21 22 23 24 25 26 27 28 29 30 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64];
%subjects = [01 02];
for subject=subjects
    
subject = num2str(subject, '%02d');
cd(['/multlearn/data/ds-mlearn/derivatives/fmriprep/sub-' subject '/func']) % Navigate to the subject's directory

for run=1:6

events = tdfread(['sub-' subject '_task-learn_run-' num2str(run) '_events.tsv'], '\t'); % Read onset times file
events.trial_type = strtrim(string(events.trial_type)); % Convert char array to string array, to make logical comparisons easier

choiceIdx = find(events.trial_type == "choice");
feedbackIdx = find(events.trial_type == "feedback");

spe = load(['/multlearn/data/fittedParameters/sub-' subject '/spe.mat']).spe;

names{1} = 'Choice';
onsets{1} = events.onset(choiceIdx);
durations{1} = events.duration(choiceIdx);
names{2} = 'Feedback';
onsets{2} = events.onset(feedbackIdx);
durations{2} = events.duration(feedbackIdx);
pmod = struct('name', {}, 'param', {}, 'poly', {});
pmod(1).name{1}  = 'Statistical';
pmod(1).param{1} = spe(run,:) - nanmean(spe(run,:));
pmod(1).poly{1}  = 1;
orth{1} = false;
orth{2} = false;



destination = fullfile(['/multlearn/data/ds-mlearn/derivatives/fmriprep/sub-' num2str(subject)], ['beh'], [model_version]);
if ~exist(destination,'dir')
    mkdir(destination)
end
save(fullfile(destination, ['/run_' num2str(run) '_conditions.mat']), 'names', 'onsets', 'durations', 'pmod', 'orth')

end
end
=======
function conditions(model_version)

subjects = [01 02 03 04 05 06 07 09 10 11 12 14 15 17 18 19 20 21 22 23 24 25 26 27 28 29 30 33 34 35 36 37 38 39 40 41 42 43 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64];
%subjects = [01 02];
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

spe = load(['/data/fittedParameters/sub-' subject '/spe.mat']).spe;

names{1} = 'Choice';
onsets{1} = events.onset(choiceIdx);
durations{1} = events.duration(choiceIdx);
names{2} = 'Feedback';
onsets{2} = events.onset(feedbackIdx);
durations{2} = events.duration(feedbackIdx);
pmod = struct('name', {}, 'param', {}, 'poly', {});
if runType == "tactile"
pmod(1).name{1}  = 'StatisticalTactile';
elseif runType == "audio"
pmod(1).name{1} = 'StatisticalAudio';
end
pmod(1).param{1} = spe(run,:) - nanmean(spe(run,:));
pmod(1).poly{1}  = 1;
orth{1} = false;
orth{2} = false;



destination = fullfile(['/data/ds-mlearn/derivatives/fmriprep/sub-' num2str(subject)], ['beh'], [model_version]);
if ~exist(destination,'dir')
    mkdir(destination)
end
save(fullfile(destination, ['/run_' num2str(run) '_conditions.mat']), 'names', 'onsets', 'durations', 'pmod', 'orth')

end
end
>>>>>>> 5f4c3248c8658a48e33ddde3a64a3c3023bbd04f
end