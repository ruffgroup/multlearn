% Discover devices connected to your system using |daqlist|. To learn more
% about an individual device, access the entry in the device table.
d = daqlist;

% Create data acquisition
dq = daq("ni");
rate = 20000;
dq.Rate = rate;
% Adding analog output channel
ch = addoutput(dq,"Dev1", "ao0","Voltage");

% Creating output tactile
A = 1;
duration = 1;
f = 50;
t = (1:duration*rate)/rate;
signal1 = A*sin(2*pi*f*t);
preload(dq, signal1')
start(dq,"repeatoutput")
% ⋮write(dq,signal')
% Device output now repeated while MATLAB continues.
pause(duration)
stop(dq)