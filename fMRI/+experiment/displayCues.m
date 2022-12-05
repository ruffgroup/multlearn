function [stimulusOnsetTime, visual, audio, tactile, combinationProb] = displayCues(trialNr, audioPair, tactilePair, modality0A, modality0B, modality0C, modality1A, modality1B, modality1C, modality2A, modality2B, modality2C, trialStructure, runNr, window, xCenter, yCenter, fixationCoordinates,fixationLineWidth, grey, black, image0, image1, image2, audioDataAll, audioTrials, dq, stimulusThreshold, yesKey, noKey, yesTick, noCross, tactile0, tactile1, tactile2, et, eyeTracking, functionalKeys, runStartTime, stimulusOnsetTimes)
    

    % Setting  accrding to the trial structure
    visual = trialStructure(trialNr,1);
    if visual == 0
        image = image0;
    elseif visual == 1
        image = image1;
    elseif visual == 2
        image = image2;
    end     

%     experiment.tickCrossDisplay(window, xCenter, yCenter, black, yesKey, yesTick, noCross, functionalKeys);
    
    image = Screen('MakeTexture', window, image);
    Screen('DrawTexture', window, image, [], [xCenter-300, yCenter-300, xCenter+300, yCenter+300]);
    Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, grey, [xCenter yCenter], 2); 

    stimulusOnsetTime = runStartTime + stimulusOnsetTimes(trialNr);
    WaitSecs('UntilTime', stimulusOnsetTime)

    % Visual cue shown
    Screen('Flip', window);
    Screen('Close', image);
        

    if eyeTracking
        et.setAnalyseMessage('Stimulus Onset');
        et.setRecordingMessage('Stimulus Onset');
    end

    % Similarly setting audio accrding to the trial structure
    if audioTrials(trialNr)
        audio = trialStructure(trialNr, 2);
        tactile = nan;

        audioData = audioDataAll{audio+1};
        pahandle = PsychPortAudio('Open', [], 1, [], 48000, 2, [], 0.015);    
        PsychPortAudio('FillBuffer', pahandle, audioData);
        PsychPortAudio('Start', pahandle);

    else
        tactile = trialStructure(trialNr, 2);
        audio = nan;
        if tactile == 0
            tactileData = tactile0;
        elseif tactile == 1
            tactileData = tactile1;
        elseif tactile == 2
            tactileData = tactile2;
        end   
        preload(dq, tactileData');
        start(dq,"repeatoutput")
    end
    
    
    % Noting down at each step what the combination probability was for the
    % two modalities based on defined modality cooccurance task structure
    if visual == 0
        if audio == audioPair(1) || tactile == tactilePair(1)
            combinationProb = modality0A;
        elseif audio == audioPair(2) || tactile == tactilePair(2)
            combinationProb = modality0B;
        elseif audio == audioPair(3) || tactile == tactilePair(3)
            combinationProb = modality0C;
        end   
    elseif visual == 1
        if audio == audioPair(1) || tactile == tactilePair(1)
            combinationProb = modality1A;
        elseif audio == audioPair(2) || tactile == tactilePair(2)
            combinationProb = modality1B;
        elseif audio == audioPair(3) || tactile == tactilePair(3)
            combinationProb = modality1C;
        end   
    elseif visual == 2
        if audio == audioPair(1) || tactile == tactilePair(1)
            combinationProb = modality2A;
        elseif audio == audioPair(2) || tactile == tactilePair(2)
            combinationProb = modality2B;
        elseif audio == audioPair(3) || tactile == tactilePair(3)
            combinationProb = modality2C;
        end    
    end
    
    
end