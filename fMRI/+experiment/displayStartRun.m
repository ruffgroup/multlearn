function displayStartRun(runNr, window,  xCenter, yCenter, yesKey, yesTick, noCross, black, functionalKeys, audioTrials)


    % Allowing response only from select number of keys.
    
    % Displaying fixation screens at the beginning of each run
    Screen('Flip', window);
    RestrictKeysForKbCheck(KbName(functionalKeys));


    if all(audioTrials)
        disp('fd')
        % Presenting the final screen to participants to indicate end of run
        StartScreen = sprintf(['You are on run %s of the experiment\n'...
            'This will be an audio-visual run.\n\n\n\n'...
            'Pay Attention and remember which buttons\n'...
            'corresponds to Attract and Not Attract!\n\n\n\n'...
            'Please press a button to start the run.'],num2str(runNr));
    else
        disp('fd')
        % Presenting the final screen to participants to indicate end of run
        StartScreen = sprintf(['You are on run %s of the experiment\n\n\n\n'...
        'This will be an visuo-tactile run.\n'...
        'Pay Attention and remember which buttons\n'...
        'corresponds to Attract and Not Attract!\n\n\n\n'...
        'Please press a button to start the run.'],num2str(runNr));
    end


    experiment.tickCrossDisplay(window, xCenter, yCenter, black, yesKey, yesTick, noCross, functionalKeys);
    

    Screen('TextSize', window, 35);
        
    Press = false;
    DrawFormattedText(window, StartScreen, 'center','center', black, [], [], [], 1.5);
    Screen('Flip', window);
    WaitSecs(0.2);
    
    while ~Press
            
        [Press, ~, KeyCode] = KbCheck();
            
        if Press
                
            break

        end
        
%         if strcmpi(KbName(KeyCode), 'escape') % Participants given option to click escape key and close screen.
%             sca;
%         end
           
    end

    Screen('TextSize', window, 35);

    % Displaying fixation screens at the beginning of each run
    Screen('Flip', window);

        
end
            
            