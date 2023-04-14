function typeChangePrompt(audioTrials, window, trialNr, black)         
    if trialNr == 1
        if audioTrials(1) == 1
            DrawFormattedText(window,'20 image-audio rounds', 'center','center', black);
            Screen('Flip', window);
            WaitSecs(2);
        elseif audioTrials(1) == 0
             DrawFormattedText(window,'20 image-touch rounds', 'center','center', black);
             Screen('Flip', window);
             WaitSecs(2);
        end
    elseif trialNr == 21
        if audioTrials(21) == 1
            DrawFormattedText(window,'20 image-audio rounds', 'center','center', black);
            Screen('Flip', window);
            WaitSecs(2);
        elseif audioTrials(21) == 0
             DrawFormattedText(window,'20 image-touch rounds', 'center','center', black);
             Screen('Flip', window);
             WaitSecs(2);
        end
    elseif trialNr == 41
        if audioTrials(41) == 1
            DrawFormattedText(window,'20 image-audio rounds', 'center','center', black);
            Screen('Flip', window);
            WaitSecs(2);
        elseif audioTrials(41) == 0
             DrawFormattedText(window,'20 image-touch rounds', 'center','center', black);
             Screen('Flip', window);
             WaitSecs(2);
        end
    end
end
    