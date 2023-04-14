function scannerTrigger(keyTrigger, window, black, runNr)


    waitingScanner = sprintf('Please wait for scanner.');

    DrawFormattedText(window, waitingScanner, 'center','center', black);
    Screen('Flip', window);

    [~, keyCode] = KbPressWait();
    keyCode = find(keyCode, 1);
    while keyCode ~= keyTrigger
       % WaitSecs(.005);
        [~, keyCode] = KbPressWait();
        keyCode = find(keyCode, 1);
    end

end
