function displayFinalScreen(runNr, window, black, nrRuns, totalReward, eyeTracking, et, runFileName, eyeDataSet)   

% Experimenter has to press spacebar at end of each run


    if eyeTracking
        et.stopRecording();
        et.closeFile();
        et.receiveFile();
        movefile( runFileName, eyeDataSet );
    end


    RestrictKeysForKbCheck(KbName('Space'));



    % Presenting the final screen to participants to indicate end of run
    finalScreenofRun = sprintf(['You have finished this run\n'...
    'Please lie still.']);
    
    % Indicating end of the experiment for the participant
    finalScreen = sprintf(['You are done with the experiment.\n'...
    'You win a total of CHF ',num2str(round(totalReward))]);

    Screen('TextSize', window, 35);
    if runNr < nrRuns
        
        Press = false;
        DrawFormattedText(window, finalScreenofRun, 'center','center', black, [], [], [], 1.5);
        Screen('Flip', window);
        WaitSecs(0.2);
        
        while ~Press
            
            [Press, ~, KeyCode] = KbCheck();
            
            if Press
                
                break
                
            end

        end
        
        if strcmpi(KbName(KeyCode), 'escape') % Participants given option to click escape key and close screen.
            sca;
        end
        
            
    end
    Screen('TextSize', window, 35);
    if runNr == nrRuns
        DrawFormattedText(window, finalScreen, 'center','center', black, [], [], [], 1.5);
        Screen('Flip', window);
        WaitSecs(1.5);
    end 

end