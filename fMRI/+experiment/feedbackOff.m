function[feedbackOffset] = feedbackOff(window, fixationCoordinates, fixationLineWidth, black, xCenter, yCenter, feedbackTime, yesKey, yesTick, noCross, et, eyeTracking, functionalKeys, feedbackOnset, feedbackOffsetTimes, trialNr, runStartTime)
        
        Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, black, [xCenter yCenter], 2); 
%         experiment.tickCrossDisplay(window, xCenter, yCenter, black, yesKey, yesTick, noCross, functionalKeys);
        
        feedbackOffset = runStartTime + feedbackOffsetTimes(trialNr);
        WaitSecs('UntilTime', feedbackOffset)
        %   start of result screen presentation
        Screen('Flip', window);
        if eyeTracking
            et.setAnalyseMessage('Feedback Offset');
            et.setRecordingMessage('Feedback Offset');
        end
        
        Screen('TextSize', window, 35);

end