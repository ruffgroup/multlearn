%% Defining major variables

% Getting the name of the computer that the participant is working on
% in the BLU lab 
% Should work everywhere but comment this line if you face problems running it anywhere else.
%[~, systemName] = system('hostname');

% Setting type of experiment. 0 for tones,1 for beeps
beeps = 1;

% Setting number of runs (runs of experiment)
nrRuns = 6; 

% Number of trials in each run. Should be a multiple of 3 and a number
% which when multiplied by each of modality0A, 0B and 0c etc gives a whole number
% With modalit0A, 0B and 0C set to 0.5, 0.35 and 0.15, the number of trials
% needed become a multiple of 60.
nrTrials = 60;

%  Creating a table to store the important values
savedValues = cell(nrRuns,1);
savedVal = table();

% If eyeTracking is TRUE then experiment will carry out eye Tracking
eyeTracking = true;

% If fMRI is TRUE then, it will wait for scanner triggers
fMRI = true;


%Clearing up the workspace and getting ready to start
close all;
clc;
sca;

%Performing standard setup of feature level 2 for Psychtoolbox
PsychDefaultSetup(2);


%Setting the display screen for the experiment as the highest screen number
%that is connected to the experimental setup. This is to project the
%experiment on the secondary computer if it is connected and only to rely
%on the primary computer if there is no secondary monitor available
allScreens = (Screen('Screens'));
screenNumber = max(allScreens);

% Settings for the audio
InitializePsychSound(1); % Initializing PsychPort audio.
allAudioDevices = PsychPortAudio('GetDevices');

% Giving participants 1.5 second + some jitter (1 sec) to respond.

responseTimeThreshold = 3.0; %+rand;
stimulusThreshold = 1.5; % Time when stimulus is displayed on screen. Response is also recorded for this
feedbackTime = 1.0;% Time till which feedback is shown

KbName('UnifyKeyNames'); % Making keyboards across Operating Systems similar


% Getting location of current directory where files are
filePath = mfilename('fullpath');
% Creating sessionFolder in order to save data
dataFolder = fullfile(fileparts(filePath),'MRIdata');
if ~exist(dataFolder,'dir')
    mkdir(dataFolder);
end
sessionFolder = fullfile(fileparts(dataFolder),['MRIdata/MS_MR_Sess',date]);
if ~exist(sessionFolder,'dir')
    mkdir(sessionFolder);
end

addpath(fullfile(fileparts(filePath),'Stimuli/Visual'));
addpath(fullfile(fileparts(filePath),'Stimuli/Auditory'));

addpath(fullfile(fileparts(filePath),'Timings'));
onsets = load('bestOnsetMat.mat').design(:,1);
jitterShuffler = randperm(6);

% FeedbackOffsets
feedbackOffsetTimes = onsets(5:4:end);
feedbackOffsetTimesMat =  reshape(feedbackOffsetTimes, [60,6]);
subElement = feedbackOffsetTimesMat(end,1:5);
feedbackOffsetTimesMat(:,2:6) = feedbackOffsetTimesMat(:,2:6) - subElement;
feedbackOffsetTimes = feedbackOffsetTimesMat(:,jitterShuffler);
feedbackOffsetTimes = feedbackOffsetTimes(:);

%stim Onsets
stimulusOnsetTimes = onsets(2:4:end);
stimulusOnsetTimesMat =  reshape(stimulusOnsetTimes, [60,6]);
% stimSubElement = stimulusOnsetTimesMat(end,1:5);
stimulusOnsetTimesMat(:,2:6) = stimulusOnsetTimesMat(:,2:6) - subElement;
stimulusOnsetTimes = stimulusOnsetTimesMat(:,jitterShuffler);
stimulusOnsetTimes = stimulusOnsetTimes(:);

% FeedbackOnsets
feedbackOnsetTimes = onsets(4:4:end);
feedbackOnsetTimesMat =  reshape(feedbackOnsetTimes, [60,6]);
% feedOnSubElement = feedbackOnsetTimesMat(end,1:5);
feedbackOnsetTimesMat(:,2:6) = feedbackOnsetTimesMat(:,2:6) - subElement;
feedbackOnsetTimes = feedbackOnsetTimesMat(:,jitterShuffler);
feedbackOnsetTimes = feedbackOnsetTimes(:);



rng('shuffle')  % Seeds random number generator based on current time.

% Defining black, white and grey for the background screens
white = WhiteIndex(screenNumber);
black = BlackIndex(screenNumber);
grey = white/1.8;
darkGrey = white/2.5;
red = [1, 0, 0];
green = [0, 100./255., 0.0];



%%   Defining additional variables   

defaultRewardProb = 0.8;
rewardMag = 1.0;
punishProb = 0.0;
punishMag = 0.0;

% How many money points earn them 1 Fr in real.
maxPerRun = 8.;
defaultMoneyFactor = maxPerRun/18.0;

% Setting up thresholds for accuracy to later assign them to easy, medium
% and hard groups
easyThreshold = 0.56;
mediumThreshold = 0.7;

%Setting up frequencies of co occurences of different modalities.
%Note that A, B and C here take the value of the awardedPair generated
%by the random permutation. So, it is not 0,0 that is always presented
%50 percent and rewarded for right response but 0,2 (lets say dicated
%by the audioPair variable) that gets presented the most and then also
%rewarded for right response.
modality0A = 0.5;
modality0B = 0.35;
modality0C = 0.15;
modality1A = 0.15;
modality1B = 0.50;
modality1C = 0.35;
modality2A = 0.35;
modality2B = 0.15;
modality2C = 0.50;


% Now the underlying visual setand the audio files within the sets are randomly
% permutated.
% Random permutation for choosing audio set to be taken along the
% visual set
visualSetPermutation = randperm(nrRuns)-1;
audioSetPermutation = randperm(ceil(nrRuns/2))-1;
% Pairing auditory within a set randomly to visual of the set
% runNr. So if the output of audioPair is 2,1,0 (A,B,C) for example - means that
% visual 0 is paired to audio 2 modality0A/3 times (50/3%), visual 0 is with
% audio 1 modality0B(35%/3) times, and 00 occurs modality0C/3 (15%/3)
% times. 
audioPair = randperm(3) - 1; % Generating the permutation for audio that will be mapped to the visual 0,1,2
tactilePair = randperm(3) - 1;



%% Build experiment information structure
experimentInfo = experiment.addInfo( ...
    'title', 'Multisensory Learning Task experiment', ...
    'Experimenter', 'Saurabh Bedi', ...
    'nrTrials', nrTrials, ...
    'expToolbox', 'psychtoolbox' ...
    );
experimentInfo.viewParam.fontSize = 35;



%% saving participant ID as enterred by experimenter

participantID = input('participantID: ', 's');
while isnan(str2double(participantID))
    disp('Please enter an integer');
    participantID = input('participantID: ', 's');
end
participantGender = input('Gender [m/f]: ', 's');
participantAge = input('Age : ', 's');
Num_scanned = input('How many times were you scanned : ', 's');

%% Opening screen and setting screen properties

%Checking whether it is a mac. if yes, then sync tests are skipped.
%Otherwise they are not. This is because mac always throws error for
%synctests. However it is otherwise extremely essential
if ismac()
    Screen('Preference', 'SkipSyncTests', 1);
end

% Settings needed for loading image later
PsychImaging('PrepareConfiguration');

% Setting size of screen through screenRect function. Would be used later,
% just naming it here to add to experimentInfo
screenRect = [0, 0, 1200, 600];

% Opening screen at max screen number
[window, screenRect] = PsychImaging('OpenWindow', screenNumber, darkGrey);


% Getting screen window size now
[screenXpixels, screenYpixels] = Screen('WindowSize', window);


%Getting coordinates for the centre of the screen window
[xCenter, yCenter] = RectCenter(screenRect);


% Setting up blend function for the screen and taking the recommended
% smoothing settigs as given by help command - 'GL_SRC_ALPHA' and 'GL_ONE_MINUS_SRC_ALPHA'
Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

%Hide cursor
HideCursor();


% Setting fixation cross parameters between cues.
fixationLineWidth = 4;
fixationCrossSize = 40;
xCoords = [-fixationCrossSize fixationCrossSize 0 0];
yCoords = [0 0 -fixationCrossSize fixationCrossSize];
fixationCoordinates = [xCoords; yCoords];


% Allowing response only from select number of keys.
keyTrigger = '5%';
% 4$ is left button inside, 2@ is the right one
functionalKeys = {'4$', '2@', 'escape', keyTrigger}; %'1!', '3#'
RestrictKeysForKbCheck(KbName(functionalKeys));

% These awarded Pairings are constant for one participant for all 3 runs
% but are assigned according to whether it is tacztile or audio trial
greenPairs = {};



% Yes tick and wrong cross
[yesTick] = imread('cue_right.png');
[noCross] = imread('cue_wrong.png');

yesTick = Screen('MakeTexture', window, yesTick);
noCross = Screen('MakeTexture', window, noCross);



%% Setting up tactile stimulator
d = daqlist;
% Create data acquisition
dq = daq("ni");
% Adding analog output channel
ch = addoutput(dq,"Dev3", "ao0","Voltage");


et = EyeTracker();
if eyeTracking
    %et = EyeTracker();
    et.init(screenNumber, screenRect, false, []);
end



%% Main Task Run

totalReward = 65.0; % Starting with a total reward value and adding more reward based on performance at end
randomAT = rand;
randATfull = rand;
for runNr = 1:nrRuns



    %% Setting up eye tracker if defined above
    if eyeTracking
        try
            et.calibrate();
        catch
            disp('skipped calibratoin');
        end
        runFileName = sprintf('M%sR%01d.edf', participantID, runNr);
        et.openFile(runFileName);
%         et.startRecording();
    end

    
    %% Setting up tactile frequencies used later
    %Defining tactile stimulation pa2ameters
    % Creating output tactile
    [tactileSeq, tactile0, tactile1, tactile2, practiceTactile0, practiceTactile1] = experiment.createTouch(runNr, dq, stimulusThreshold);

    
    
    % Loading all trial structures that are counterbalanced for every 20 trials
    load('counterbalancedTrialStructure.mat');
    sampleNumber = randi(length(counterbalancedTrialStructure));
    trialStructure = counterbalancedTrialStructure(:,:,sampleNumber);
    
    if randomAT <= 0.5 % randomizing this with rand as well
        % Defining whether it is audio-visual or visuo-tactile trial
        if runNr == 1 || runNr == 3 || runNr == 5
        audioTrials = repelem([1;1;1],20);
        tactileTrials = ~audioTrials;
        elseif runNr == 2 || runNr == 4 || runNr == 6
           audioTrials = repelem([0;0;0],20);
           tactileTrials = ~audioTrials;
        end
    else
        if runNr == 1 || runNr == 3 || runNr == 5
           audioTrials = repelem([0;0;0],20);
           tactileTrials = ~audioTrials;
        elseif runNr == 2 || runNr == 4 || runNr == 6
           audioTrials = repelem([1;1;1],20);
           tactileTrials = ~audioTrials;
        end
    end
    % Till here we defined the layout of audio and tactile
    
    
    if runNr == 1 || runNr == 2
        audio0Data = audioread(['audio',num2str(audioSetPermutation(1)*3 + 0),'.wav']);
        audio0Data = [audio0Data'; audio0Data']; % transposition
        audio1Data = audioread(['audio',num2str(audioSetPermutation(1)*3 + 1),'.wav']);
        audio1Data = [audio1Data'; audio1Data']; % transposition
        audio2Data = audioread(['audio',num2str(audioSetPermutation(1)*3 + 2),'.wav']);
        audio2Data = [audio2Data'; audio2Data']; % transposition
        if runNr == 1
            image0 = imread(['insect',num2str(visualSetPermutation(1)*3 + 0),'.png']);
            image1 = imread(['insect',num2str(visualSetPermutation(1)*3 + 1),'.png']);
            image2 = imread(['insect',num2str(visualSetPermutation(1)*3 + 2),'.png']);
        elseif runNr == 2
            image0 = imread(['insect',num2str(visualSetPermutation(2)*3 + 0),'.png']);
            image1 = imread(['insect',num2str(visualSetPermutation(2)*3 + 1),'.png']);
            image2 = imread(['insect',num2str(visualSetPermutation(2)*3 + 2),'.png']);
        end
    % Using the same audio stimuli for all
    elseif runNr == 3 || runNr == 4
        audio0Data = audioread(['audio',num2str(audioSetPermutation(2)*3 + 0),'.wav']);
        audio0Data = [audio0Data'; audio0Data']; % transposition
        audio1Data = audioread(['audio',num2str(audioSetPermutation(2)*3 + 1),'.wav']);
        audio1Data = [audio1Data'; audio1Data']; % transposition
        audio2Data = audioread(['audio',num2str(audioSetPermutation(2)*3 + 2),'.wav']);
        audio2Data = [audio2Data'; audio2Data']; % transposition
        if runNr == 3
            image0 = imread(['insect',num2str(visualSetPermutation(3)*3 + 0),'.png']);
            image1 = imread(['insect',num2str(visualSetPermutation(3)*3 + 1),'.png']);
            image2 = imread(['insect',num2str(visualSetPermutation(3)*3 + 2),'.png']);
        elseif runNr == 4
            image0 = imread(['insect',num2str(visualSetPermutation(4)*3 + 0),'.png']);
            image1 = imread(['insect',num2str(visualSetPermutation(4)*3 + 1),'.png']);
            image2 = imread(['insect',num2str(visualSetPermutation(4)*3 + 2),'.png']); 
        end
        
    elseif runNr == 5 || runNr == 6
        audio0Data = audioread(['audio',num2str(audioSetPermutation(3)*3 + 0),'.wav']);
        audio0Data = [audio0Data'; audio0Data']; % transposition
        audio1Data = audioread(['audio',num2str(audioSetPermutation(3)*3 + 1),'.wav']);
        audio1Data = [audio1Data'; audio1Data']; % transposition
        audio2Data = audioread(['audio',num2str(audioSetPermutation(3)*3 + 2),'.wav']);
        audio2Data = [audio2Data'; audio2Data']; % transposition
        if runNr == 5
            image0 = imread(['insect',num2str(visualSetPermutation(5)*3 + 0),'.png']);
            image1 = imread(['insect',num2str(visualSetPermutation(5)*3 + 1),'.png']);
            image2 = imread(['insect',num2str(visualSetPermutation(5)*3 + 2),'.png']);
        elseif runNr == 6
            image0 = imread(['insect',num2str(visualSetPermutation(6)*3 + 0),'.png']);
            image1 = imread(['insect',num2str(visualSetPermutation(6)*3 + 1),'.png']);
            image2 = imread(['insect',num2str(visualSetPermutation(6)*3 + 2),'.png']); 
        end
    else
        print("You need to generate more stimuli");
    end

    audioDataAll = {audio0Data, audio1Data, audio2Data};
        

    % rewardProb1 and rewardProb2 are already assigned at the end of runs 1
    % and 2 respectively
     if runNr == 1 || runNr == 2
        rewardProb = defaultRewardProb;
        moneyFactor = defaultMoneyFactor;
     elseif runNr == 3 || runNr == 5
        rewardProb = rewardProb1;
        moneyFactor = moneyFactor1;
     elseif runNr == 4 || runNr == 6    
      rewardProb = rewardProb2;
      moneyFactor = moneyFactor2;
     end
    
    
    % These awarded Pairings are constant for one participant for all 3
    % runs
    greenPairs(audioTrials==1,1) = {[0,audioPair(1)]};
    greenPairs(audioTrials==1,2) = {[1,audioPair(2)]};
    greenPairs(audioTrials==1,3) = {[2,audioPair(3)]};
    audioGreenPairs = {[0,audioPair(1)], [1,audioPair(2)], [2,audioPair(3)]};
    greenPairs(tactileTrials==1,1) = {[0,tactilePair(1)]};
    greenPairs(tactileTrials==1,2) ={[1,tactilePair(2)]};
    greenPairs(tactileTrials==1,3) ={[2,tactilePair(3)]};
    tactileGreenPairs = {[0,tactilePair(1)], [1,tactilePair(2)], [2,tactilePair(3)]};

    % Now randomizing the trial structure based on whether it is audio or
    % tactile trials
    trialStructure(trialStructure(:,2)=="A" & audioTrials,2) = audioPair(1);
    trialStructure(trialStructure(:,2)=="B" & audioTrials,2) = audioPair(2);
    trialStructure(trialStructure(:,2)=="C" & audioTrials,2) = audioPair(3);
    trialStructure(trialStructure(:,2)=="A" & tactileTrials,2) = tactilePair(1);
    trialStructure(trialStructure(:,2)=="B" & tactileTrials,2) = tactilePair(2);
    trialStructure(trialStructure(:,2)=="C" & tactileTrials,2) = tactilePair(3);

    trialStructure = double(trialStructure);
  

    %Defining how long the tempFeedbackAccuracy down below should be. A
    %length of 10 means that there are rewardProb number of rewards given in each 10
    %correct trials. A value of 5 means, it is better and less confounded
    %as then they would get rewardProb number of rewards for each 5
    %accurate choices. This way, the probability of not giving a reward
    %even when correct is maintained at higher resolution (by using lower
    %value for rewardStructureLength). However, note that decreasing this
    %value of rewardStructureLength does not let us choose certain reward
    %probabilities.. for example, a rewardStructureLength = 10, lets us use
    %any number from 1 to 10. A rewardStructureLength of 5 only lets us
    %chose rewardProbs of 0.2,0.4,0.66,0.8 or 1.0 (we cannot chose 0.9 for
    %ex)
    rewardStructureLength = 10; 
    
    % Generating the reward structures for both when they are accurate and when
    % they are not. This is generated for each session seperately
    tempFeedbackAccuracy = repelem([1,0], int64([rewardProb*rewardStructureLength, (1-rewardProb)*rewardStructureLength]));
    tempFeedbackAccuracy = tempFeedbackAccuracy(randperm(length(tempFeedbackAccuracy)));
    feedbackAccuracy = [];
    for i=1:(nrTrials/(length(tempFeedbackAccuracy)))
        feedbackAccuracy = [feedbackAccuracy tempFeedbackAccuracy(randperm(length(tempFeedbackAccuracy)))];
    end
    
   

    % Third type of counterbalancing so that all visual stimuli get equal
    % number of wrong feedbacks
    while sum(trialStructure(logical(~feedbackAccuracy),1)==0) ~= int8((1-rewardProb)*nrTrials/3) | sum(trialStructure(logical(~feedbackAccuracy),1)==1) ~= int8((1-rewardProb)*nrTrials/3) | sum(trialStructure(logical(~feedbackAccuracy),1)==2) ~= int8((1-rewardProb)*nrTrials/3)
        feedbackAccuracy = [];
        for i=1:(nrTrials/(length(tempFeedbackAccuracy)))
            feedbackAccuracy = [feedbackAccuracy tempFeedbackAccuracy(randperm(length(tempFeedbackAccuracy)))];
        end
    end

    % Initializing the tracking index of how much they were correct in
    % every 5 trials. Gets initialized in each session.
    accurateIndex = 1;
    inaccurateIndex = 1;

    % Setting up properties for the text displayed.
    Screen('TextSize', window, experimentInfo.viewParam.fontSize);
    Screen('TextFont', window, 'Arial');

    if rem(str2double(participantID), 2) ~= 0
        yesKey = functionalKeys(2);
        noKey = functionalKeys(1);

    else
        yesKey = functionalKeys(1);
        noKey = functionalKeys(2);
    end

   % Start screen giving optoin to participant
    experiment.displayStartRun(runNr, window,  xCenter, yCenter, yesKey, yesTick, noCross, black, functionalKeys, audioTrials)


    % Creating a temporary table to store the important values in each run
    temporarySavedValues = table; 


    if eyeTracking
        et.startRecording();
        et.setRecordingMessage(sprintf('MultisensoryLearning Run %d', runNr));
    end
    
    % with the start of each run
    if fMRI
        experiment.scannerTrigger(keyTrigger, window, black, runNr)
    end

    runStartTime = GetSecs;  

    for trialNr = 1:nrTrials

        % Displaying fixation, note that response time is calculated
        % fro this fixation scereen showing time.
        [trialStartTime] = experiment.displayPreFixation(window, xCenter, yCenter, fixationCoordinates,fixationLineWidth, white, black, yesKey, noKey, yesTick, noCross, functionalKeys, runStartTime);


        % responseTimeOverYet has value false when threshold for response time
        % has not reached
        responseTimeOverYet = false; 

        % Displaying Cues
        [stimulusOnsetTime, visual, audio, tactile, combinationProb] = experiment.displayCues(trialNr, audioPair, tactilePair, modality0A, modality0B, modality0C, modality1A, modality1B, modality1C, modality2A, modality2B, modality2C, trialStructure, runNr, window, xCenter, yCenter, fixationCoordinates,fixationLineWidth, grey, black, image0, image1, image2, audioDataAll, audioTrials, dq, stimulusThreshold, yesKey, noKey, yesTick, noCross, tactile0, tactile1, tactile2, et, eyeTracking, functionalKeys, runStartTime, stimulusOnsetTimes);

         %Collecting time and identity of keyboard response
        [responseTimeOverYet, responsePressTime, KeyCode, IsKeyDown, stimulusOffsetTime, responseTimeOver] = experiment.keyboardResponse(runNr, responseTimeOverYet, stimulusOnsetTime, responseTimeThreshold, window, fixationCoordinates,fixationLineWidth, black, grey, xCenter, yCenter, visual, stimulusThreshold, image0, image1, image2, dq, yesKey, noKey, yesTick, noCross, et, eyeTracking, functionalKeys, runStartTime);
        if responseTimeOverYet
            responseTime = nan;
        else
            responseTime = (responsePressTime - stimulusOnsetTime); % Noted from start of fixation screen
        end

        % Displaying reward slide
        [feedbackOnset, correctResponse, reward] = experiment.displayReward(window, trialNr, greenPairs, rewardMag, ...
            rewardProb, punishMag, visual, audio, tactile, KeyCode,fixationCoordinates, fixationLineWidth, black, grey, xCenter, yCenter, IsKeyDown, audioTrials, feedbackTime, feedbackAccuracy, participantID, yesKey, noKey, yesTick, noCross, et, eyeTracking, functionalKeys, runStartTime, responseTimeOver, feedbackOnsetTimes, runNr);

        %Saving all important response information temporarily in a table
        %for each run
        [temporarySavedValues] = experiment.accumulatingRunData(trialNr, runNr, correctResponse, responseTimeOverYet, KeyCode, combinationProb, responseTimeThreshold, temporarySavedValues, visual, audio, tactile, reward, stimulusOnsetTime, stimulusOffsetTime, trialStartTime, responsePressTime, responseTime, responseTimeOver, feedbackOnset, runStartTime, audioPair, tactilePair);

        totalAccuracy = nansum(temporarySavedValues.accurate)/sum(temporarySavedValues.chosenKey ~= "");

        temporarySavedValues.totalAccuracy(trialNr,1) = totalAccuracy; 
        temporarySavedValues.totalErrorRate(trialNr,1) = 1 - totalAccuracy;


        % Delete this later
        temporarySavedValues.accurateIndex(trialNr,1) = accurateIndex;
        temporarySavedValues.inaccurateIndex(trialNr,1) = inaccurateIndex;


        % Updating the accurate index which is used to run through the
        % matrix of accurateReward and then give rewards probabilistically.
        % Same for inaccurateIndex
        if temporarySavedValues.accurate(trialNr,1) == 1
            accurateIndex = accurateIndex + 1;
        elseif isnan(temporarySavedValues.accurate(trialNr,1))
            inaccurateIndex = inaccurateIndex + 0;
        else
            inaccurateIndex = inaccurateIndex + 1;
        end
        
        temporarySavedValues.feedbackAccuracy(trialNr,1) = feedbackAccuracy(trialNr);


        %Setting feedbackOffset here once all computatoins are done and
        %lags are taken care off
        [feedbackOffset] = experiment.feedbackOff(window, fixationCoordinates, fixationLineWidth, black, xCenter, yCenter, feedbackTime, yesKey, yesTick, noCross, et, eyeTracking, functionalKeys, feedbackOnset, feedbackOffsetTimes, trialNr, runStartTime, runNr);

        temporarySavedValues.feedbackOffsetTime(trialNr,1) = feedbackOffset-runStartTime;

    end

    % Assigning what rewardProb and moneyFactor participants get assigned
    % in later odd and even runs based on calibration runs
    if runNr == 1
        acc1 = nansum(temporarySavedValues.accurate)/nrTrials;
        if acc1 < easyThreshold
            rewardProb1 = 0.9;
            moneyFactor1 = (maxPerRun/24.)*90./100.0;
         elseif acc1 >= easyThreshold && acc1 < mediumThreshold
            rewardProb1 = 0.8;
            moneyFactor1 = (maxPerRun/18.)*95./100.0;
         else
            rewardProb1 = 0.7;
            moneyFactor1 = (maxPerRun/12.)*100./100.0;
        end

    elseif runNr == 2  
        acc2 = nansum(temporarySavedValues.accurate)/nrTrials;
         if acc2 < easyThreshold
            rewardProb2 = 0.9;
            moneyFactor2 = (maxPerRun/24.)*90./100.0;
         elseif acc2 >= easyThreshold && acc2 < mediumThreshold
            rewardProb2 = 0.8; 
            moneyFactor2 = (maxPerRun/18.)*95./100.0;
         else
            rewardProb2 = 0.7;
            moneyFactor2 = (maxPerRun/12.)*100./100.0;
         end
    end



    % Making a cell array and inputing different runs for a participant in
    % that
    savedValues{runNr,1} = temporarySavedValues;  
    totalReward = totalReward + (sum(savedValues{runNr,1}.reward(~isnan(savedValues{runNr,1}.reward)))-nrTrials*rewardMag/2)*moneyFactor;
    if totalReward<65
        totalReward = 65;
    end


    eyeDataSet = fullfile(fileparts(sessionFolder),['/MS_MR_Sess',date],participantID,'/ETdata');
    if ~exist(eyeDataSet,'dir')
         mkdir(eyeDataSet);
    end


    %Displaying final screen of the run
    experiment.displayFinalScreen(runNr, window, black, nrRuns, totalReward, eyeTracking, et, runFileName, eyeDataSet)   

    savedVal = [savedVal; temporarySavedValues];  
    dataSet = fullfile(fileparts(sessionFolder),['/MS_MR_Sess',date],participantID);
    if ~exist(dataSet,'dir')
         mkdir(dataSet);
    end


    % Saving payment file
    fid = fopen([participantID,'ID',', pay = ',num2str(round(totalReward)),'.txt'],'wt');
    fprintf(fid, 'Payment = %d, rounded payment = %d', totalReward, round(totalReward));
    fclose(fid);

    % Opening csv and .mat file formats to save in
    expInfoCSV = sprintf('participant%s_expInfo.csv', participantID);
    savedValuesCSV = sprintf('participant%s_savedValues.csv', participantID);
    filename = sprintf('participant%s.xlsx', participantID);
    filenameMat = sprintf('participant%s.mat', participantID);


    % Writing to individual sheets in excel after each run
    writetable(savedValues{runNr,1},fullfile(dataSet)+"/"+filename,'Sheet',runNr+1);
    writetable(savedVal, fullfile(dataSet)+"/"+savedValuesCSV);
    save(fullfile(dataSet)+"/"+filenameMat, 'savedVal');

    
end
%     stop(player);

%% Adding information to structure experimentInfo
experimentInfo.participantID = participantID;
experimentInfo.participantGender = participantGender;
experimentInfo.participantNumOfTimesScanned = Num_scanned;
experimentInfo.nrRuns = nrRuns;
experimentInfo.nrTrials = nrTrials;
experimentInfo.date = date;
experimentInfo.audioPair = audioPair;
experimentInfo.tactilePair = tactilePair;
experimentInfo.modality0AFrequency = modality0A;
experimentInfo.modality0BFrequency = modality0B;
experimentInfo.modality0CFrequency = modality0C;
experimentInfo.modality1AFrequency = modality1A;
experimentInfo.modality1BFrequency = modality1B;
experimentInfo.modality1CFrequency = modality1C;
experimentInfo.modality2AFrequency = modality2A;
experimentInfo.modality2BFrequency = modality2B;
experimentInfo.modality2CFrequency = modality2C;
experimentInfo.rewardMagnitude = rewardMag;
experimentInfo.rewardProbability1 = rewardProb1;
experimentInfo.rewardProbability2 = rewardProb2;
experimentInfo.audioGreenPairs = audioGreenPairs; 
experimentInfo.tactileGreenPairs = tactileGreenPairs;
experimentInfo.nrRuns = nrRuns;
experimentInfo.allScreens = allScreens;
experimentInfo.responseTimeThreshold = responseTimeThreshold;
experimentInfo.feedbackTime = feedbackTime;
experimentInfo.moneyFactor1 = moneyFactor1;
experimentInfo.moneyFactor2 = moneyFactor2;
experimentInfo.easyThreshold = easyThreshold;
experimentInfo.mediumThreshold = mediumThreshold;
experimentInfo.beeps = beeps;
experimentInfo.visualSetPermutation = visualSetPermutation;
experimentInfo.audioSetPermutation = audioSetPermutation;
experimentInfo.jitterShuffler = jitterShuffler;


% Showing input box to participants
%[participantAge] = experiment.getParticipant(window, screenXpixels, screenYpixels, darkGrey, black);
experimentInfo.participantAge = participantAge;


PsychPortAudio('Close');
Screen('CloseAll')
ShowCursor();

experimentInfo = struct2table(repmat(experimentInfo,2,1)); %Done to be able to convert struct2table later as it does not work with 1,1 structures
% Writing the details of experiment Info in the first excel sheet.
writetable(experimentInfo(1,:),fullfile(dataSet)+"/"+filename,'Sheet',1);
save(fullfile(dataSet)+"/"+filenameMat, 'savedVal', 'experimentInfo');
writetable(experimentInfo(1,:), fullfile(dataSet)+"/"+expInfoCSV);
writetable(savedVal, fullfile(dataSet)+"/"+savedValuesCSV);
 
experimentInfo = experimentInfo(1,:);
% As all important variables have been saved 
% in relevant structures, we clear the irrelevant variables from the
% workspace.
clearvars -except experimentInfo savedValues;
        