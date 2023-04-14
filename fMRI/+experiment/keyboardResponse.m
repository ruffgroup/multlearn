function [responseTimeOverYet, responsePressTime, KeyCode, IsKeyDown, stimulusOffsetTime, responseTimeOver] = keyboardResponse(runNr, responseTimeOverYet, stimulusOnsetTime, responseTimeThreshold, window, fixationCoordinates,fixationLineWidth, black, grey, xCenter, yCenter, visual, stimulusThreshold, image0, image1, image2, dq, yesKey, noKey, yesTick, noCross, et, eyeTracking, functionalKeys, runStartTime)

    % Collecting keyboard data until response time threshold has passed. If
    % the threshold is crossed then the participants are simply shown the
    % next cue.
    
    i = 0;       
    


    while ~responseTimeOverYet
       
        [IsKeyDown, responsePressTime, KeyCode] = KbCheck;

        if IsKeyDown % Conditional to break from the even handler while loop as soon as key is clicked.
                % Changing color of fixation cross when response made

            if strcmpi(KbName(KeyCode), 'escape') % Participants given option to click escape key and close screen.
                sca;
            end
                
            break
            
        else
            
            % If response is not made yet after stimulus has been removed
            % it keeps entering this conditional and only shows the grey
            % fixation till responseTime is over
            if (responsePressTime - stimulusOnsetTime) >= stimulusThreshold
               
                stop(dq)  
                i = i+1;
                if i == 1

                    if eyeTracking
                        et.setAnalyseMessage('Stimulus Offset');
                        et.setRecordingMessage('Stimulus Offset');
                    end 

%                     experiment.tickCrossDisplay(window, xCenter, yCenter, black, yesKey, yesTick, noCross, functionalKeys);
                    Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, grey, [xCenter yCenter], 2); 
                    stimulusOffsetTime = Screen('Flip', window); 

                end   
               
            end
            
        end

       % Conditional to break out of while loop if key is not clicked in threshold time.
        if responsePressTime - stimulusOnsetTime >= responseTimeThreshold
            responseTimeOverYet = true;
        end
            
    end
        
    % if the key was pressed within response time threshold
    if IsKeyDown

%         experiment.tickCrossDisplay(window, xCenter, yCenter, black, yesKey, yesTick, noCross, functionalKeys);

        % If response is made when stimulus was on the screen
        if (responsePressTime - stimulusOnsetTime) < stimulusThreshold
            
            if visual == 0
               image = Screen('MakeTexture', window, image0);
            elseif visual == 1
               image = Screen('MakeTexture', window, image1);  
            else
               image = Screen('MakeTexture', window, image2);
            end
            Screen('DrawTexture', window, image, [], [xCenter-300, yCenter-300, xCenter+300, yCenter+300]);
            
            Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, black, [xCenter yCenter], 2); 
            Screen('Flip', window);

            timeElapsed = GetSecs - stimulusOnsetTime;
            WaitSecs(stimulusThreshold - (responsePressTime - stimulusOnsetTime) - timeElapsed);
            % stimulus Offset
            stop(dq)

            stimulusOffsetTime = GetSecs;

%             experiment.tickCrossDisplay(window, xCenter, yCenter, black, yesKey, yesTick, noCross, functionalKeys);
            Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, black, [xCenter yCenter], 2); 
            Screen('Flip', window);
            

            if eyeTracking
                et.setAnalyseMessage('Stimulus Offset');
                et.setRecordingMessage('Stimulus Offset');
            end

        else
            Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, black, [xCenter yCenter], 2); 
            Screen('Flip', window); 
            
        end 
    end
    timeElapsed = GetSecs-stimulusOffsetTime;
    WaitSecs(responseTimeThreshold - stimulusThreshold - timeElapsed);
    responseTimeOver = GetSecs;
    % If key is not pressed within threshold for response time
%     experiment.tickCrossDisplay(window, xCenter, yCenter, black, yesKey, yesTick, noCross, functionalKeys);
    Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, black, [xCenter yCenter], 2);
    Screen('Flip', window);

end