function [tactileSeq, tactile0, tactile1, tactile2, practiceTactile0, practiceTactile1] = createTouch(runNr, dq, stimulusThreshold)   
    %Defining tactile stimulation paramters
    % Creating output tactile
    rate = 48000;
    dq.Rate = rate;
    A = 3;

    duration = stimulusThreshold;
  %  t = (1:duration*rate)/rate;
    
   
    beepLength = 0.015;

    nBeeps = 8 + 1;
    beepTotal = beepLength * nBeeps;
    waitTotal = duration - beepTotal;
    waitMin = waitTotal/22;
   

    if runNr == 1 || runNr == 2
        
        waitTimes1 = waitMin.*[1, 1, 5,  3,  1, 8,  1, 1];
        waitTimes2 = waitMin.*[5.25, 5.25, 0.15, 0.45, 0.15, 0.25, 10.25, 0.25];
        waitTimes3 = waitMin.*[2, 4, 6, 8, 0.5, 0.25, 0.15, 0.15];

    elseif runNr == 3 || runNr == 4
 
        waitTimes1 = waitMin.*[6, 0.25, 1, 4, 2, 0.25, 2, 5]; 
        waitTimes2 = waitMin.*[2, 5, 2, 2, 2, 2, 5, 2];  
        waitTimes3 = waitMin.*[0.20, 0.20, 10.25, 10.25, 0.20, 0.25, 0.25, 0.25];        
   
    else 
        
        waitTimes1 = waitMin.*[10, 5, 1, 1, 1, 1, 1, 5];
        waitTimes2 = waitMin.*[2.75, 2.75, 2.75, 2.0, 2.0, 2.75, 2.75, 2.75];
        waitTimes3 = waitMin.*[0.5, 1, 1.5, 2, 2.5, 3.5, 4.5, 6.5];

    end


    tactile0 = experiment.transformTouch(waitTimes1, beepLength, rate, A, duration);
    tactile1 = experiment.transformTouch(waitTimes2, beepLength, rate, A, duration);
    tactile2 = experiment.transformTouch(waitTimes3, beepLength, rate, A, duration);


    %% instructions sounds

    waitTimes15 = waitMin.*[4, 0.5, 2, 4, 2, 4, 4, 0.5];
    waitTimes15 = flip(waitTimes15);
    practiceTactile0 = experiment.transformTouch(waitTimes15, beepLength, rate, A, duration);


    waitTimes16 = waitMin.*[9, 0.5, 0.2, 4, 0.2, 4, 1, 0.5];
    practiceTactile1 = experiment.transformTouch(waitTimes16, beepLength, rate, A, duration);
    
    tactileSeq = [tactile0; tactile1; tactile2; practiceTactile0; practiceTactile1];

    