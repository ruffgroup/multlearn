function tickCrossDisplay(window, xCenter, yCenter, black, yesKey, yesTick, noCross, functionalKeys)

key1 = 'Index Finger';
key2 = 'Middle Finger';
if strcmp(string(yesKey), functionalKeys{2})
    key1 = 'Middle Finger';
    key2 = 'Index Finger';
end



yesKeyDisplay = sprintf(['Attract - \n', key1]);
noKeyDisplay = sprintf(['Not Attract - \n', key2]);
Screen('TextSize', window, 35);

rightText = yesKeyDisplay;
rightTexture = yesTick;
leftText = noKeyDisplay;
leftTexture = noCross;

if strcmp(yesKey,functionalKeys{1})
    rightText = noKeyDisplay;
    rightTexture = noCross;
    leftText = yesKeyDisplay;
    leftTexture = yesTick;
end



DrawFormattedText(window, rightText, xCenter+550, yCenter-145, black, [], [], [], 1.5);
Screen('DrawTexture', window, rightTexture, [], [xCenter+550, yCenter-65, xCenter+700, yCenter+65]);
DrawFormattedText(window, leftText, xCenter - 700, yCenter - 145, black, [], [], [], 1.5);
Screen('DrawTexture', window, leftTexture, [], [xCenter-700, yCenter-65, xCenter-550, yCenter+65]);

end