function [y] = transformTouch(waitTimes, beepLength, rate, A, duration)

    freq  = 100;
    beepStart = beepLength + waitTimes;
    beepStart = [0, beepStart];
    
    y = zeros(1,duration*rate);
    time = zeros(1,duration*rate);

    for i = 1:(length(beepStart)-1)
        beepStart(i+1) = beepStart(i) + beepStart(i+1);
    end


    for i = 1:duration*rate

        time(i) = i/rate;
        
        j = max(find(beepStart<time(i)));

        % finding out if the time(i) lies in the interval of the beeps or not
        if any(((time(i)>beepStart) - (time(i)<beepStart + beepLength)==0)==1) 
            y(1,i) = A .* sin(2 * pi * freq*(time(i)-beepStart(j)));

        else
            y(i) = 0;


        end

    end
end