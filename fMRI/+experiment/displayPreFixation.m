function[trialStartTime] = displayPreFixation(window, xCenter, yCenter, fixationCoordinates,fixationLineWidth, white, black, yesKey, noKey, yesTick, noCross, functionalKeys, runStartTime)
 
    Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, black, [xCenter, yCenter], 2); 
%     experiment.tickCrossDisplay(window, xCenter, yCenter, black, yesKey, yesTick, noCross, functionalKeys);
    trialStartTime = Screen('Flip', window);
 
end