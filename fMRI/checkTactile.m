%% Setting up tactile stimulator
d = daqlist;
% Create data acquisition
dq = daq("ni");
% Adding analog output channel
ch = addoutput(dq,"Dev3", "ao0","Voltage");
stimulusThreshold = 1.5;
runNr = 1;
% Creating output tactile
[tactileSeq, tactile0, tactile1, tactile2, practiceTactile0, practiceTactile1] = experiment.createTouch(runNr, dq, stimulusThreshold);

preload(dq, practiceTactile0');
tic
start(dq,"repeatoutput")
while toc <= 1.5
    % just something
end
stop(dq)  