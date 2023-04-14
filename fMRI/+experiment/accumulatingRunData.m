function [temporarySavedValues] = accumulatingRunData(trialNr, runNr, correctResponse, responseTimeOverYet, KeyCode, combinationProb, responseTimeThreshold, temporarySavedValues, visual, audio, tactile, reward, stimulusOnsetTime, stimulusOffsetTime, trialStartTime, responsePressTime, responseTime, responseTimeOver, feedbackOnset, runStartTime, audioPair, tactilePair)
        
        % Saving relevant response data
        temporarySavedValues.trialNumber(trialNr,1) = trialNr; % First column as trial number
    
        temporarySavedValues.runNumber(trialNr,1) = runNr;
                
        temporarySavedValues.visualSet(trialNr,1) = runNr-1; % Equal to the run number - 1 as that is how visual set is chosen.
        
       % temporarySavedValues.audioSet(trialNr,1) = audioSetPermutation(runNr);

        temporarySavedValues.audioPair(trialNr,1) = {audioPair};
       
        temporarySavedValues.tactilePair(trialNr,1) = {tactilePair};

        temporarySavedValues.visual(trialNr,1) = visual;
        
        temporarySavedValues.audio(trialNr,1) = audio;

        temporarySavedValues.tactile(trialNr,1) = tactile;

        temporarySavedValues.combinationConditionalProbability(trialNr,1) = combinationProb;
        
        temporarySavedValues.correctResponse(trialNr,1) = correctResponse; % Saving presented cue value as correct response
         

        % Saving the clicked key value by the participant.
        if responseTimeOverYet
            chosenKey = "";
            temporarySavedValues.chosenKey(trialNr,1) = chosenKey; 
            
        end

        if ~responseTimeOverYet
            chosenKey = string(KbName(KeyCode));
            temporarySavedValues.chosenKey(trialNr,1) = chosenKey; 
        end
   
    
        %Saving whether participant chose the correct response or not in
        %variable accurate

        accurate = double(temporarySavedValues.correctResponse(trialNr,1) == temporarySavedValues.chosenKey(trialNr,1));
        if temporarySavedValues.chosenKey(trialNr,1) == ""
            accurate = NaN;
        end

        
        temporarySavedValues.accurate(trialNr,1) = accurate; % Saving accurate in the table temporarySavedValues
   
        temporarySavedValues.reward(trialNr,1) = reward; % Did participant get rewarded
                
        % Saving response time of the participants.
        temporarySavedValues.runStartTime(trialNr,1) = runStartTime;
        temporarySavedValues.trialStartTime(trialNr,1) = trialStartTime-runStartTime;
        temporarySavedValues.stimulusOnsetTime(trialNr,1) = stimulusOnsetTime-runStartTime;
        temporarySavedValues.stimulusOffsetTime(trialNr,1) = stimulusOffsetTime-runStartTime;
        temporarySavedValues.responsePressTime(trialNr,1) = responsePressTime-runStartTime;
        % Saving all sorts of times of participants
        temporarySavedValues.responseTime(trialNr,1) = responseTime;
        temporarySavedValues.responseTimeOver(trialNr,1) = responseTimeOver-runStartTime;
        temporarySavedValues.feedbackOnsetTime(trialNr,1) = feedbackOnset-runStartTime;

        
        temporarySavedValues.responseTimeThreshold(trialNr,1) = responseTimeThreshold;

end