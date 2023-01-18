function[feedbackOnset, correctResponse, reward] = displayReward(window, trialNr, greenPairs, rewardMag, rewardProb, punishMag, visual, audio, tactile, KeyCode, fixationCoordinates, fixationLineWidth, black, grey, xCenter, yCenter, IsKeyDown, audioTrials, feedbackTime, feedbackAccuracy, participantID, yesKey, noKey, yesTick, noCross, et, eyeTracking, functionalKeys, runStartTime, responseTimeOver, feedbackOnsetTimes, runNr)
 
    
    rewardSlide = '+1';    
    noRewardSlide = '0';
    noResponseSlide = '?';
    
%     experiment.tickCrossDisplay(window, xCenter, yCenter, black, yesKey, yesTick, noCross, functionalKeys);

   
    size = 140;
    Screen('TextSize', window, size);
    
    
%     Screen('DrawLines', window, fixationCoordinates,fixationLineWidth, black, [xCenter yCenter], 2); 

    if audioTrials(trialNr)
        secondModality = audio;
    else
        secondModality = tactile;
    end

    

    % Checking if the given (visaul,audio) or (visual,tactile) pair is one of the green pairs
    % as depending on that the response will be rewarded
    if any(cellfun(@(x) isequal(x, [visual,secondModality]), greenPairs(trialNr,:))) 
        
        if rem(str2double(participantID), 2) ~= 0
            correctResponse = functionalKeys(2);   % Marking right arrow as the correct response when the awarded Pair is the same as 
            incorrectResponse = functionalKeys(1);
        else
            correctResponse = functionalKeys(1);
            incorrectResponse = functionalKeys(2);
        end

        if strcmp(correctResponse,KbName( KeyCode)) % Means they were accurate
       
            reward = feedbackAccuracy(trialNr); % Probabilistic reward
            % Showing reward one on screen
            if reward == 1 % Happens with rewardProb
                DrawFormattedText(window, rewardSlide, xCenter -135./250.*size, yCenter+90./250.*size, black, [], [], [], 1.5);
            else
                % Showing non reward one on screen
                DrawFormattedText(window, noRewardSlide, xCenter -68./250.*size, yCenter+90./250.*size, black, [], [], [], 1.5);
            end
                

            
        elseif strcmp(incorrectResponse,KbName(KeyCode))% Conditional on when they were not accurate for when correct answer was functionalKeys(2)
                  
            reward = double(~feedbackAccuracy(trialNr));
            if reward == 0 % Happens with rewardProb
                % Showing non reward one on screen
                DrawFormattedText(window, noRewardSlide, xCenter -68./250.*size, yCenter+90./250.*size, black, [], [], [], 1.5);
            else
                % Showing non reward one on screen. Happens with 1 - rewardProb
                DrawFormattedText(window, rewardSlide, xCenter -135./250.*size, yCenter+90./250.*size, black, [], [], [], 1.5);
            end

        else
                reward = NaN;
                DrawFormattedText(window, noResponseSlide, xCenter -68./250.*size, yCenter+90./250.*size, black, [], [], [], 1.5);
            
        end 
            
        
    else
        
        if rem(str2double(participantID), 2) ~= 0
            correctResponse = functionalKeys(1); % making left arrow as correct response if the cell presented was different from awarded cells
            incorrectResponse = functionalKeys(2);
        else
            correctResponse = functionalKeys(2);
            incorrectResponse = functionalKeys(1);
            
        end

        if strcmp(correctResponse,KbName(KeyCode))
                        
            reward = feedbackAccuracy(trialNr); % Probabilistic reward for accurate choice
            % Showing reward one on screen
            if reward == 1 % Happens with rewardProb
                DrawFormattedText(window, rewardSlide, xCenter -135./250.*size, yCenter+90./250.*size, black, [], [], [], 1.5);
            else
                % Showing non reward one on screen
                DrawFormattedText(window, noRewardSlide, xCenter -68./250.*size, yCenter+90./250.*size, black, [], [], [], 1.5);
            end
                

            
        elseif strcmp(incorrectResponse,KbName( KeyCode))
            
            reward = double(~feedbackAccuracy(trialNr));
            if reward == 0 % Happens with rewardProb
                % Showing non reward one on screen
                DrawFormattedText(window, noRewardSlide, xCenter -68./250.*size, yCenter+90./250.*size, black, [], [], [], 1.5);
            else
                % Showing non reward one on screen. Happens with 1 - rewardProb
                DrawFormattedText(window, rewardSlide, xCenter -135./250.*size, yCenter+90./250.*size, black, [], [], [], 1.5);
            end

        else
            reward = NaN;
            DrawFormattedText(window, noResponseSlide, xCenter -68./250.*size, yCenter+90./250.*size, black, [], [], [], 1.5);          
        end 

    end

    feedbackOnset = runStartTime + feedbackOnsetTimes(trialNr+(runNr-1)*60);
    WaitSecs('UntilTime', feedbackOnset)

    % start of result screen presentation
    if eyeTracking
        et.setAnalyseMessage('Feedback Onset');
        et.setRecordingMessage('Feedback Onset');
    end
    Screen('Flip', window);