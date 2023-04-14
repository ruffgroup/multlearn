% Instructions for ODD NUMBERED participants; RIGHT Attracts

close all;
clc;
sca;

PsychDefaultSetup(2);

allScreens = (Screen('Screens'));
screenNumber = max(allScreens);

% Settings for the audio
InitializePsychSound(1); % Initializing PsychPort audio.
allAudioDevices = PsychPortAudio('GetDevices');

KbName('UnifyKeyNames'); % Making keyboards across Operating Systems similar

filePath = mfilename('fullpath');
addpath(fullfile(fileparts(filePath),'Stimuli/Visual'));
addpath(fullfile(fileparts(filePath),'Stimuli/Auditory'));

rng('shuffle')  % Seeds random number generator based on current time.

white = WhiteIndex(screenNumber);
black = BlackIndex(screenNumber);
grey = white/1.8;
green = [0, 0.109, 1.0];
darkGrey = white/2.5;

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

yesKey = "RightArrow";
noKey = "LeftArrow";

Screen('TextSize', window, 35);

rewardSlide = sprintf(['+1']);
noRewardSlide = sprintf(['0']);
noResponseSlide = sprintf(['?']);

functionalKeys = {'RightArrow', 'LeftArrow', 'escape'};
RestrictKeysForKbCheck(KbName(functionalKeys));

% defining variables
responseTimeThreshold = 3; %+rand;
stimulusThreshold = 1.5; % Time when stimulus is displayed on screen. Response is also recorded for this
responseTimeOverYet = false;


%Image cues

%% Defining all print statements and importing variables

welcomeScreen = sprintf(['Welcome to the experiment! Thank you for participating.\n\n'...
    'We will now go over the instructions.\n'...
    'Please read each instruction screen carefully, as you will not be\n'...
    'able to go back to previous screens.\n\n\n']);


practiceRound1 = sprintf(['To start, we will give you an overview of the different rounds.\n\n'...
    'After that, you will play a few practice sound rounds.\n'...
    'The points you win during the practice rounds will not be counted towards your final score.\n\n'...
    'Note that the insect images and sounds will also differ\n'...
    'from the ones you will encounter in the real rounds.\n\n']);

RoundIntroOne = sprintf(['All the rounds proceed as follows:\n\n'...
    '1. First, you see a black cross in the middle of the screen. \n\n\n']);

practiceCross = sprintf('This is what the cross in the middle looks like:\n\n\n');


RoundIntroTwo = sprintf(['All the rounds proceed as follows:\n\n'...
    '1. First, you see a black cross in the middle of the screen. \n\n'...
    '2. Then, the cross changes color from black to grey. \n\n\n']);

practiceCrossChange = sprintf(['This is the grey colored cross (see below): \n\n'...
    'Almost immediately after this you will see the insect and hear the sound,\n'...
    'or you will see the insect and feel the touch pattern.\n\n\n']);


RoundIntroThree = sprintf(['All the rounds proceed as follows:\n\n'...
    '1. First, you see a black cross in the middle of the screen. \n\n'...
    '2. Then, the cross changes color from black to grey. \n\n'...
    '3. Next, you see an insect and simultaneously hear a sound or feel a touch.\n'...
    'From this point on you can make a choice by pressing the left or right arrow.\n\n\n']);


practiceRoundVisualCues = sprintf(['These are the two insects\n'...
    'that you will see during practice.\n\n'...
    'Notice how the patterns on the insects are different.\n\n\n']);

practiceRoundAuditoryCues = sprintf(['These are the two sounds\n'...
    'that you will hear during practice.\n\n\n']);

RoundIntroFour = sprintf(['All the rounds proceed as follows:\n\n'...
    '1. First, you see a black cross in the middle of the screen. \n\n'...
    '2. Then, the cross changes color from black to grey. \n\n'...
    '3. Next, you see an insect and simultaneously hear a sound or feel a touch.\n'...
    'From this point on you can make a choice by pressing the left or right arrow.\n\n'...
    '4. Finally, the insect goes away and the cross reappears.\n'...
    'The color of the cross depends on whether you already made a choice (black) \n'...
    'or you still need to make one (grey). \n\n\n']);

RoundIntroFive = sprintf(['All the rounds proceed as follows:\n\n'...
    '1. First, you see a black cross in the middle of the screen. \n\n'...
    '2. Then, the cross changes color from black to grey. \n\n'...
    '3. Next, you see an insect and simultaneously hear a sound or feel a touch.\n'...
    'From this point on you can make a choice by pressing the left or right arrow.\n\n'...
    '4. Finally, the insect goes away and the cross reappears.\n'...
    'The color of the cross depends on whether you already made a choice (black) \n'...
    'or you still need to make one (grey). \n\n'...
    '5. You get a reward (+1 point) if your prediction was correct.\n'...
    'If you make an incorrect prediction you get nothing (0 points).\n'...
    'Finally, you see a (?) if you did not respond in time. \n\n\n']);


RewardIntro = sprintf(['This is the +1 reward you will see when you correctly predicted\n'...
    'that the given mating call or dance attracted (or did not attract) a partner.\n\n'...
    'i.e. If you press the right arrow to indicate the mating call/dance\n'...
    'WAS successful when in fact the insect DID attract a partner,\n'...
    'or press the left arrow to indicate that the mating call/dance \n'...
    'WAS NOT successful when indeed they DID NOT attract a partner, \n'...
    'you will get +1 point. \n\n\n']);

NoRewardIntro = sprintf(['This is the 0 reward you will see when your prediction was incorrect,\n'...
    'either by pressing the right arrow to indicate the mating call/dance WAS successful\n'...
    'when in fact it WAS NOT, or by pressing the left arrow to indicate\n'...
    'that they DID NOT attract a partner when in fact they DID.\n\n\n']);

NoResponseIntro = sprintf(['If you did not respond in the given time,\n'...
    'you will ALWAYS see the "?" as shown below. \n\n\n']);


AllTogether = sprintf(['Now, let us practice the "audio-image" rounds.\n'...
    'To make the practice rounds easier the insects will ALWAYS attract a partner when using\n'...
    'their specific sounds, and NEVER when imitating a different insect.\n\n'...
    'The points you win now will not be counted towards your final score.\n\n']);


practiceRightCorrect = sprintf(['You pressed the right arrow.\n'...
    'Thus, you believe that the insect attracted a partner with this mating call.\n\n'...
    'You also saw the +1 at the end of the run, indicating a correct choice.\n'...
    'This means that this mating call did indeed attract a partner,\n'...
    'and is specific to this species.\n'...
    'You get +1 point for this round.\n\n\n']);

practiceLeftIncorrect = sprintf(['You pressed the left arrow.\n'...
    'Thus, you believe that the insect did not attract a partner with this mating call.\n\n'...
    'You also saw the 0 at the end of the round, indicating an incorrect choice.\n'...
    'This means that this mating call did attract a partner,\n'...
    'and is specific to this species.\n'...
    'You win 0 points for this round.\n\n'...
    'The correct answer would have been the right arrow as the insect\n'...
    'did in fact attract a partner. \n\n\n']);



practiceNoSelect = sprintf(['You did not choose in the allotted time!\n'...
    'You only have a fixed amount of time to respond (%s seconds).\n\n'...
    'If you do not press the left or right arrow in the given time,\n\n'...
    'you loose the chance to get a point for that round.\n\n\n'], string(responseTimeThreshold));


instructionFive = sprintf(['You need to respond before the cross in the middle goes away.\n\n'...
    'You will hear the sound and see the image of the insect for %s seconds\n'...
    'after the cross changes color.\n\n'...
    'Then, you will get an additional %s seconds to respond.\n'...
    'You can respond as soon as the insect is shown\n'...
    'and as long as the cross is shown (total %s sec).\n\n\n'], string(stimulusThreshold), string(responseTimeThreshold - stimulusThreshold), string(responseTimeThreshold));


practiceRound2 = sprintf(['Let us play another practice round! \n\n'....
    'Remember that these insect images and sounds are different from the ones\n'...
    'you will encounter later in the real rounds.\n\n\n']);


practiceRightIncorrect = sprintf(['You pressed the right arrow.\n'...
    'Thus, you believe that the insect attracted a partner with this mating call.\n\n'...
    'You also saw the 0 at the end of the round, indicating an incorrect choice.\n'...
    'This means that this mating call did not attract a partner,\n'...
    'and is not specific to this species.\n'...
    'You win 0 points for this round.\n\n'...
    'The correct answer would have been the left arrow as the insect\n'...
    'did not attract a partner. \n\n\n']);

practiceLeftCorrect = sprintf(['You pressed the left arrow.\n'...
    'Thus, you believe that the insect did not attract a partner with this mating call.\n\n'...
    'You also saw the +1 at the end of the run, indicating a correct choice.\n'...
    'This means that this mating call did indeed not attract a partner,\n'...
    'and is not specific to this species.\n'...
    'You get +1 point for this round.\n\n\n']);



instructionSixPart1 = sprintf(['Now it is time to move to the scanner!\n\n\n'...
    'The real rounds will be slightly more difficult than the practice rounds. \n\n']);
instructionSixPart2 = sprintf(['In the actual task, the specific mating calls and dances will NOT always be successful.\n'...
    'Instead, they will be successful MOST of the time,\n'...
    'meaning that there is a chance that they do not attract a partner and you get 0 points \n'...
    'even when the mating call or dance was specific to the insect.\n'...
    'Similarly, if the mating call or dance was NOT specific to the species,\n'...
    '(i.e. they imitated a different species),\n'...
    'there is also a chance that they DO attract a partner..\n\n']);
instructionSixPart3 = sprintf(['This makes it harder to figure out the specific image-sound or image-touch pairs,\n'...
    'so you will need a few rounds to be certain of the most likely prediction for each pairing. \n\n'...
    'If you get 0 points in a round,\n'...
    'it does not necessarily mean that the insect was imitating a different species. \n'...
    'Similarly, winning +1 in a round does not guarantee\n'...
    'that the call/dance was specific to the insect.\n\n']); 


summarySlide = sprintf(['SUMMARY\n\n\n'...
    '1. You get combinations of insect images and sounds, or of images and touch patterns.\n\n'...
    '2. You then predict whether the insect will attract a partner (Right Arrow) or not (Left).\n\n'...
    '3. If your prediction was correct, you will get rewarded.\n\n'...
    '3. If your prediction was incorrect, you will not get rewarded.\n\n'...
    '4. Your job is to learn which combinations are specific to the insect\n'...
    'and thus most likely to attract a partner, so that you can win rewards\n'...
    'by having correct predictions most of the time.\n\n'...
    '5. In the end, you get paid based on how many points you got above chance.\n\n\n']);




%% Welcome screen

DrawFormattedText(window, welcomeScreen, 'center','center', black, [], [], [], 1.5);
DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
Screen('Flip', window);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);



%% practice cue 1

% Practice Image cue 1
[practiceImage1] = imread('insect21.png');
% Practice Image cue 2
[practiceImage2] = imread('insect23.png');

% Practice Audio cue 1
[practiceAudioData1] = audioread('audio13.wav');
practiceAudioData1 = [practiceAudioData1'; practiceAudioData1']; % transposition
% Practice Audio cue 2
[practiceAudioData2] = audioread('audio14.wav');
practiceAudioData2 = [practiceAudioData2'; practiceAudioData2']; % transposition


% Run
DrawFormattedText(window, practiceRound1, 'center','center', black, [], [], [], 1.5);
DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
Screen('Flip', window);
WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);


% INTRO ONE
DrawFormattedText(window, RoundIntroOne, 'center', 0.1*screenYpixels, black, [], [], [], 1.5);
DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
Screen('Flip', window);
WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);


% Cross
DrawFormattedText(window, practiceCross, 'center', 0.1*screenYpixels, black, [], [], [], 1.5);
Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, black, [xCenter yCenter], 2);
DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
Screen('Flip', window);
WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);


% INTRO TWO
DrawFormattedText(window, RoundIntroTwo, 'center',0.1*screenYpixels, black, [], [], [], 1.5);
DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
Screen('Flip', window);
WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);


% Cross change
DrawFormattedText(window, practiceCrossChange, 'center',0.1*screenYpixels, black, [], [], [], 1.5);
Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, grey, [xCenter yCenter], 2);
DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
Screen('Flip', window);
WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);


% INTRO THREE
DrawFormattedText(window, RoundIntroThree, 'center',0.1*screenYpixels, black, [], [], [], 1.5);
DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
Screen('Flip', window);
WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);


% Practice visual cues
DrawFormattedText(window, practiceRoundVisualCues, 'center',0.1*screenYpixels, black, [], [], [], 1.5);
practiceImageShow1 = Screen('MakeTexture', window, practiceImage1);
Screen('DrawTexture', window, practiceImageShow1, [], [xCenter-800, yCenter-200, xCenter-200, yCenter+200]);
practiceImageShow2 = Screen('MakeTexture', window, practiceImage2);
Screen('DrawTexture', window, practiceImageShow2, [], [xCenter+200, yCenter-200, xCenter+800, yCenter+200]);
DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
Screen('Flip', window);
WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);


% INTRO FOUR
DrawFormattedText(window, RoundIntroFour, 'center',0.1*screenYpixels, black, [], [], [], 1.5);
DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
Screen('Flip', window);
WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);

% INTRO FIVE
DrawFormattedText(window, RoundIntroFive, 'center',0.1*screenYpixels, black, [], [], [], 1.5);
DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
Screen('Flip', window);
WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);



DrawFormattedText(window, RewardIntro, 'center', 0.1*screenYpixels, black, [], [], [], 1.5);
Screen('TextSize', window, 190);
DrawFormattedText(window, rewardSlide, 'center', 0.7*screenYpixels, black, [], [], [], 1.5);
Screen('Flip', window);
Screen('TextSize', window, 35);
WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);

DrawFormattedText(window, NoRewardIntro, 'center', 0.1*screenYpixels, black, [], [], [], 1.5);
Screen('TextSize', window, 190);
DrawFormattedText(window, noRewardSlide,  'center', 0.7*screenYpixels, black, [], [], [], 1.5);
Screen('Flip', window);
Screen('TextSize', window, 35);
WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);


DrawFormattedText(window, NoResponseIntro, 'center', 0.1*screenYpixels, black, [], [], [], 1.5);
Screen('TextSize', window, 190);
DrawFormattedText(window, noResponseSlide,  'center', 0.7*screenYpixels, black, [], [], [], 1.5);
Screen('Flip', window);
Screen('TextSize', window, 35);
WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);


% All together
DrawFormattedText(window, AllTogether, 'center', 'center', black, [], [], [], 1.5);
Screen('Flip', window);
WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);


% Showing practice audio cues screen
DrawFormattedText(window, practiceRoundAuditoryCues, 'center',0.15*screenYpixels, black, [], [], [], 1.5);
Screen('Flip', window);
WaitSecs(2.0);


DrawFormattedText(window, practiceRoundAuditoryCues, 'center',0.15*screenYpixels, black, [], [], [], 1.5);
DrawFormattedText(window, 'Practice insect Sound 1', 'center',screenYpixels*0.3, black, [], [], [], 1.5);
Screen('Flip', window);
pahandle = PsychPortAudio('Open', [], 1, [], 48000, 2, [], 0.015);
PsychPortAudio('FillBuffer', pahandle, practiceAudioData1);
PsychPortAudio('Start', pahandle);
WaitSecs(3);
DrawFormattedText(window, 'Practice insect Sound 1', 'center',screenYpixels*0.3, black, [], [], [], 1.5);
Screen('Flip', window);


DrawFormattedText(window, practiceRoundAuditoryCues, 'center',0.15*screenYpixels, black, [], [], [], 1.5);
DrawFormattedText(window, 'Practice insect Sound 1', 'center',screenYpixels*0.3, black, [], [], [], 1.5);
DrawFormattedText(window, 'Practice insect Sound 2', 'center',screenYpixels*0.5, black, [], [], [], 1.5);
Screen('Flip', window);
pahandle = PsychPortAudio('Open', [], 1, [], 48000, 2, [], 0.015);
PsychPortAudio('FillBuffer', pahandle, practiceAudioData2);
PsychPortAudio('Start', pahandle);
WaitSecs(3);


DrawFormattedText(window, practiceRoundAuditoryCues, 'center',0.15*screenYpixels, black, [], [], [], 1.5);
DrawFormattedText(window, 'Practice insect Sound 1', 'center',screenYpixels*0.3, black, [], [], [], 1.5);
DrawFormattedText(window, 'Practice insect Sound 2', 'center',screenYpixels*0.5, black, [], [], [], 1.5);
DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
Screen('Flip', window);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);


%% Practice cue 1
stimulusOnsetTime = GetSecs;

Screen('TextSize', window, 35);


practiceImageShow1 = Screen('MakeTexture', window, practiceImage1);
Screen('DrawTexture', window, practiceImageShow1, [], [xCenter-300, yCenter-300, xCenter+300, yCenter+300]);
Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, grey, [xCenter yCenter], 2);
[VBLTimestamp] = Screen('Flip', window);
pahandle = PsychPortAudio('Open', [], 1, [], 48000, 2, [], 0.015);
PsychPortAudio('FillBuffer', pahandle, practiceAudioData1);
PsychPortAudio('Start', pahandle);

% response
while responseTimeOverYet == false
    [IsKeyDown, responsePressTime, KeyCode] = KbCheck;
    if IsKeyDown % Conditional to break from the even handler while loop as soon as key is clicked.
        % Changing color of fixation cross when response made
        break
    else
        if stimulusThreshold < (responsePressTime - stimulusOnsetTime)
            Screen('TextSize', window, 35);
           
            
            Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, grey, [xCenter yCenter], 2);
            [VBLTimestamp] = Screen('Flip', window);
        end
    end
    % Conditional to break out of while loop if key is not clicked in threshold time.
    if responsePressTime - stimulusOnsetTime >= responseTimeThreshold
        responseTimeOverYet = true;
    end
    if strcmpi(KbName(KeyCode), 'escape') % Participants given option to click escape key and close screen.
        sca;
    end
end

Screen('TextSize', window, 35);


if IsKeyDown
    if stimulusThreshold >= (responsePressTime - stimulusOnsetTime)
        Screen('DrawTexture', window, practiceImageShow1, [], [xCenter-300, yCenter-300, xCenter+300, yCenter+300]);
        Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, black, [xCenter yCenter], 2);
        [VBLTimestamp] = Screen('Flip', window);
        WaitSecs(stimulusThreshold - (responsePressTime - stimulusOnsetTime));
    end
    Screen('TextSize', window, 35);
    
        
    Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, black, [xCenter yCenter], 2);
    [VBLTimestamp] = Screen('Flip', window);
    WaitSecs(responseTimeThreshold - stimulusThreshold);
end



Screen('TextSize', window, 35);

    

Screen('TextSize', window, 190);
% if IsKeyDown
%     Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, black, [xCenter yCenter], 2);
% else
%     Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, grey, [xCenter yCenter], 2);
% end
correctResponse = "RightArrow";
Screen('TextSize', window, 35);

    


Screen('TextSize', window, 140);
if strcmp(correctResponse,KbName( KeyCode))
    DrawFormattedText(window, rewardSlide, xCenter -75.6, yCenter + 50.4, black, [], [], [], 1.5);
elseif strcmp("LeftArrow",KbName( KeyCode))
    DrawFormattedText(window, noRewardSlide, xCenter -38, yCenter + 50.4, black, [], [], [], 1.5);
else
    DrawFormattedText(window, noResponseSlide, xCenter -38, yCenter + 50.4, black, [], [], [], 1.5);
end
Screen('Flip', window);
WaitSecs(1.0);


Screen('TextSize', window, 35);
if IsKeyDown
    if strcmp(correctResponse,KbName( KeyCode))
        DrawFormattedText(window, practiceRightCorrect, 'center','center', black, [], [], [], 1.5);
        DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
        Screen('Flip', window);
    else
        DrawFormattedText(window, practiceLeftIncorrect, 'center','center', black, [], [], [], 1.5);
        DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
        Screen('Flip', window);
    end
else
    DrawFormattedText(window, practiceNoSelect, 'center','center', black, [], [], [], 1.5);
    DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
    Screen('Flip', window);

end

WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);
responseTimeOverYet = false;



%% Showing instruction five screen
DrawFormattedText(window, instructionFive, 'center','center', black, [], [], [], 1.5);
DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
Screen('Flip', window);
WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);
responseTimeOverYet = false;


%% Cue 3

DrawFormattedText(window, practiceRound2, 'center','center', black, [], [], [], 1.5);
DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
Screen('Flip', window);
WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);

Screen('TextSize', window, 35);

    




stimulusOnsetTime = GetSecs;
practiceImageShow2 = Screen('MakeTexture', window, practiceImage2);
Screen('DrawTexture', window, practiceImageShow2, [], [xCenter-300, yCenter-300, xCenter+300, yCenter+300]);
Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, grey, [xCenter yCenter], 2);
[VBLTimestamp] = Screen('Flip', window);
pahandle = PsychPortAudio('Open', [], 1, [], 48000, 2, [], 0.015);
PsychPortAudio('FillBuffer', pahandle, practiceAudioData1);
PsychPortAudio('Start', pahandle);

% response
while responseTimeOverYet == false
    [IsKeyDown, responsePressTime, KeyCode] = KbCheck;
    if IsKeyDown % Conditional to break from the even handler while loop as soon as key is clicked.
        % Changing color of fixation cross when response made
        break
    else
        if stimulusThreshold < (responsePressTime - stimulusOnsetTime)
            Screen('TextSize', window, 35);
            
                
            
            Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, grey, [xCenter yCenter], 2);
            [VBLTimestamp] = Screen('Flip', window);
        end
    end
    % Conditional to break out of while loop if key is not clicked in threshold time.
    if responsePressTime - stimulusOnsetTime >= responseTimeThreshold
        responseTimeOverYet = true;
    end
    if strcmpi(KbName(KeyCode), 'escape') % Participants given option to click escape key and close screen.
        sca;
    end
end


Screen('TextSize', window, 35);

    

if IsKeyDown
    if stimulusThreshold >= (responsePressTime - stimulusOnsetTime)
        Screen('DrawTexture', window, practiceImageShow2, [], [xCenter-300, yCenter-300, xCenter+300, yCenter+300]);
        Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, black, [xCenter yCenter], 2);
        [VBLTimestamp] = Screen('Flip', window);
        WaitSecs(stimulusThreshold - (responsePressTime - stimulusOnsetTime));
    end
    Screen('TextSize', window, 35);
    
       
    
    Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, black, [xCenter yCenter], 2);
    [VBLTimestamp] = Screen('Flip', window);
    WaitSecs(responseTimeThreshold - stimulusThreshold);
end


Screen('TextSize', window, 35);

   

Screen('TextSize', window, 190);
% if IsKeyDown
%     Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, black, [xCenter yCenter], 2);
% else
%     Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, grey, [xCenter yCenter], 2);
% end
correctResponse = "LeftArrow";
Screen('TextSize', window, 35);

    


Screen('TextSize', window, 140);
if strcmp(correctResponse,KbName( KeyCode))
    DrawFormattedText(window, rewardSlide, xCenter -75.6, yCenter + 50.4, black, [], [], [], 1.5);
elseif strcmp("RightArrow",KbName( KeyCode))
    DrawFormattedText(window, noRewardSlide, xCenter -38, yCenter + 50.4, black, [], [], [], 1.5);
else
    DrawFormattedText(window, noResponseSlide, xCenter -38, yCenter + 50.4, black, [], [], [], 1.5);
end
Screen('Flip', window);
WaitSecs(1.0);
Screen('TextSize', window, 35);

%
if IsKeyDown
    if strcmp(correctResponse,KbName( KeyCode))
        DrawFormattedText(window, practiceLeftCorrect, 'center','center', black, [], [], [], 1.5);
        DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
        Screen('Flip', window);
    else
        DrawFormattedText(window, practiceRightIncorrect, 'center','center', black, [], [], [], 1.5);
        DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
        Screen('Flip', window);
    end
else
    DrawFormattedText(window, practiceNoSelect, 'center','center', black, [], [], [], 1.5);
    DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
    Screen('Flip', window);

end
WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);
responseTimeOverYet = false;


%% Cue 3

DrawFormattedText(window, practiceRound2, 'center','center', black, [], [], [], 1.5);
DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
Screen('Flip', window);
WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);

Screen('TextSize', window, 35);

   




stimulusOnsetTime = GetSecs;
practiceImageShow2 = Screen('MakeTexture', window, practiceImage2);
Screen('DrawTexture', window, practiceImageShow2, [], [xCenter-300, yCenter-300, xCenter+300, yCenter+300]);
Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, grey, [xCenter yCenter], 2);
[VBLTimestamp] = Screen('Flip', window);
pahandle = PsychPortAudio('Open', [], 1, [], 48000, 2, [], 0.015);
PsychPortAudio('FillBuffer', pahandle, practiceAudioData2);
PsychPortAudio('Start', pahandle);

% response
while responseTimeOverYet == false
    [IsKeyDown, responsePressTime, KeyCode] = KbCheck;
    if IsKeyDown % Conditional to break from the even handler while loop as soon as key is clicked.
        % Changing color of fixation cross when response made
        break
    else
        if stimulusThreshold < (responsePressTime - stimulusOnsetTime)
            Screen('TextSize', window, 35);
            
                
            
            Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, grey, [xCenter yCenter], 2);
            [VBLTimestamp] = Screen('Flip', window);
        end
    end
    % Conditional to break out of while loop if key is not clicked in threshold time.
    if responsePressTime - stimulusOnsetTime >= responseTimeThreshold
        responseTimeOverYet = true;
    end
    if strcmpi(KbName(KeyCode), 'escape') % Participants given option to click escape key and close screen.
        sca;
    end
end


Screen('TextSize', window, 35);

   


if IsKeyDown
    if stimulusThreshold >= (responsePressTime - stimulusOnsetTime)
        Screen('DrawTexture', window, practiceImageShow2, [], [xCenter-300, yCenter-300, xCenter+300, yCenter+300]);
        Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, black, [xCenter yCenter], 2);
        [VBLTimestamp] = Screen('Flip', window);
        WaitSecs(stimulusThreshold - (responsePressTime - stimulusOnsetTime));
    end
    Screen('TextSize', window, 35);
    
        
    Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, black, [xCenter yCenter], 2);
    [VBLTimestamp] = Screen('Flip', window);
    WaitSecs(responseTimeThreshold - stimulusThreshold);
end


Screen('TextSize', window, 35);

    
Screen('TextSize', window, 140);
% if IsKeyDown
%     Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, black, [xCenter yCenter], 2);
% else
%     Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, grey, [xCenter yCenter], 2);
% end
correctResponse = "RightArrow";
Screen('TextSize', window, 35);

    


Screen('TextSize', window, 140);
if strcmp(correctResponse,KbName( KeyCode))
    DrawFormattedText(window, rewardSlide, xCenter -75.6, yCenter + 50.4, black, [], [], [], 1.5);
elseif strcmp("LeftArrow",KbName( KeyCode))
    DrawFormattedText(window, noRewardSlide, xCenter -38, yCenter + 50.4, black, [], [], [], 1.5);
else
    DrawFormattedText(window, noResponseSlide, xCenter -38, yCenter + 50.4, black, [], [], [], 1.5);
end
Screen('Flip', window);
WaitSecs(1.0);
Screen('TextSize', window, 35);

%
if IsKeyDown
    if strcmp(correctResponse,KbName( KeyCode))
        DrawFormattedText(window, practiceRightCorrect, 'center','center', black, [], [], [], 1.5);
        DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
        Screen('Flip', window);
    else
        DrawFormattedText(window, practiceLeftIncorrect, 'center','center', black, [], [], [], 1.5);
        DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
        Screen('Flip', window);
    end
else
    DrawFormattedText(window, practiceNoSelect, 'center','center', black, [], [], [], 1.5);
    DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
    Screen('Flip', window);

end
WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);
responseTimeOverYet = false;


%% Cue 4
stimulusOnsetTime = GetSecs;

Screen('TextSize', window, 35);




practiceImageShow1 = Screen('MakeTexture', window, practiceImage1);
Screen('DrawTexture', window, practiceImageShow1, [], [xCenter-300, yCenter-300, xCenter+300, yCenter+300]);
Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, grey, [xCenter yCenter], 2);
[VBLTimestamp] = Screen('Flip', window);
pahandle = PsychPortAudio('Open', [], 1, [], 48000, 2, [], 0.015);
PsychPortAudio('FillBuffer', pahandle, practiceAudioData2);
PsychPortAudio('Start', pahandle);

% response
while responseTimeOverYet == false
    [IsKeyDown, responsePressTime, KeyCode] = KbCheck;
    if IsKeyDown % Conditional to break from the even handler while loop as soon as key is clicked.
        % Changing color of fixation cross when response made
        break
    else
        if stimulusThreshold < (responsePressTime - stimulusOnsetTime)
            Screen('TextSize', window, 35);
            
                
            
            Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, grey, [xCenter yCenter], 2);
            [VBLTimestamp] = Screen('Flip', window);
        end
    end
    % Conditional to break out of while loop if key is not clicked in threshold time.
    if responsePressTime - stimulusOnsetTime >= responseTimeThreshold
        responseTimeOverYet = true;
    end
    if strcmpi(KbName(KeyCode), 'escape') % Participants given option to click escape key and close screen.
        sca;
    end
end

Screen('TextSize', window, 35);

   

if IsKeyDown
    if stimulusThreshold >= (responsePressTime - stimulusOnsetTime)
        Screen('DrawTexture', window, practiceImageShow1, [], [xCenter-300, yCenter-300, xCenter+300, yCenter+300]);
        Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, black, [xCenter yCenter], 2);
        [VBLTimestamp] = Screen('Flip', window);
        WaitSecs(stimulusThreshold - (responsePressTime - stimulusOnsetTime));
    end
    Screen('TextSize', window, 35);
    
        
    
    Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, black, [xCenter yCenter], 2);
    [VBLTimestamp] = Screen('Flip', window);
    WaitSecs(responseTimeThreshold - stimulusThreshold);
end



Screen('TextSize', window, 35);

    
Screen('TextSize', window, 190);
% if IsKeyDown
%     Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, black, [xCenter yCenter], 2);
% else
%     Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, grey, [xCenter yCenter], 2);
% end
correctResponse = "LeftArrow";
Screen('TextSize', window, 35);

    
Screen('TextSize', window, 140);
if strcmp(correctResponse,KbName( KeyCode))
    DrawFormattedText(window, rewardSlide, xCenter -75.6, yCenter + 50.4, black, [], [], [], 1.5);
elseif strcmp("RightArrow",KbName( KeyCode))
    DrawFormattedText(window, noRewardSlide, xCenter -38, yCenter + 50.4, black, [], [], [], 1.5);
else
    DrawFormattedText(window, noResponseSlide, xCenter -38, yCenter + 50.4, black, [], [], [], 1.5);
end
Screen('Flip', window);
WaitSecs(1.0);


Screen('TextSize', window, 35);
if IsKeyDown
    if strcmp(correctResponse,KbName( KeyCode))
        DrawFormattedText(window, practiceLeftCorrect, 'center','center', black, [], [], [], 1.5);
        DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
        Screen('Flip', window);
    else
        DrawFormattedText(window, practiceRightIncorrect, 'center','center', black, [], [], [], 1.5);
        DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
        Screen('Flip', window);
    end
else
    DrawFormattedText(window, practiceNoSelect, 'center','center', black, [], [], [], 1.5);
    DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
    Screen('Flip', window);

end

WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);
responseTimeOverYet = false;


%% Showing instruction six screen
DrawFormattedText(window, instructionSixPart1, 'center','center', black, [], [], [], 1.5);
DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
Screen('Flip', window);
WaitSecs(0.8); % I want them to look at this window properly
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);

DrawFormattedText(window, instructionSixPart2, 'center','center', black, [], [], [], 1.5);
DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
Screen('Flip', window);
WaitSecs(0.8); % I want them to look at this window properly
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);

DrawFormattedText(window, instructionSixPart3, 'center','center', black, [], [], [], 1.5);
DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
Screen('Flip', window);
WaitSecs(0.8); % I want them to look at this window properly
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);



%% Showing summary slide screen
DrawFormattedText(window, summarySlide, 'center','center', black, [], [], [], 1.5);
DrawFormattedText(window, '[Press Left or Right Arrow to continue]', 'center', 0.9*screenYpixels, green);
Screen('Flip', window);
WaitSecs(0.2);
[secs, KeyCode] = KbWait();
if strcmpi(KbName(KeyCode), 'escape')
    sca;
end
WaitSecs(0.2);
            
PsychPortAudio('Close');
Screen('CloseAll')
ShowCursor();

            