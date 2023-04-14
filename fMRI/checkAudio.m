
% Getting location of current directory where files are
filePath = mfilename('fullpath');
addpath(fullfile(fileparts(filePath),'Stimuli/Auditory'));

% Settings for the audio
InitializePsychSound(1); % Initializing PsychPort audio.
allAudioDevices = PsychPortAudio('GetDevices');

practiceAudioData = audioread(['audio',num2str(14),'.wav']);
practiceAudioData = [practiceAudioData'; practiceAudioData']; % transposition

pahandle = PsychPortAudio('Open', [], 1, [], 48000, 2, [], 0.015);    
PsychPortAudio('FillBuffer', pahandle, practiceAudioData);
PsychPortAudio('Start', pahandle);
