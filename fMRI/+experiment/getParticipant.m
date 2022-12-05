function [participantAge] = getParticipant(window, screenXpixels, screenYpixels, darkGrey, black)

%%Opening a dialog box that the participant can enter in!

    Screen('TextSize', window, 45);
    msg = 'Please input your age (in integer numbers) and click enter - ';
    [participantAge] = GetEchoString(window,msg,screenXpixels/5.5, screenYpixels/2,black,darkGrey);
    
    % Setting defaults for participantID in case experimenter mistakenly forgets to input
    % This would return a decimal number to indicate that the experimenter
    % mistakenly forgot to input this number.
    
    Screen('TextSize', window, 45);
    while str2double(participantAge) ~= abs(floor(str2double(participantAge))) || str2double(participantAge) ~= real(str2double(participantAge))
        msg = sprintf(['Not a valid age, Please input your age (in integer numbers) and click enter - ']);
        [participantAge] = GetEchoString(window,msg,screenXpixels/8.5, screenYpixels/2,black,darkGrey);
    end

    
end