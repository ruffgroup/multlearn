%% example displaying gaze, heatmap, pupil size and triggers
% Author: Marc Biedermann
% Date: 10.07.2020

% Add Edf2mat toolbox
%addpath(fullfile('/', 'pathTo@Edf2MatFolder'));


%% define variables
% data variables
dataPath = fullfile(fileparts(mfilename('fullpath')), 'MRIdata', 'MR Pilot Data', 'MS_MR_Sess14-Nov-2022','1', 'ETdata');
subject  = [1];
run      = [4];

% --> check code line 26 and 27, for the layout of the file path to the edf files


%% load EDF's
oldFolder = cd(dataPath);
edf = cell(numel(subject), numel(run));
subjCount = 1;
for currentSubj = subject
    runCount  = 1;
    for currentRun = run
        % if multiple subject folders next to each other, add to fullfile:
        %   fullfile(sprintf('%.2d', currentSubj), sprintf('myexp_%d.edf', currentRun))
        edf{subjCount, runCount} = Edf2Mat(sprintf('M%sR%01d.edf', string(currentSubj), currentRun));
        runCount = runCount + 1;
    end
    subjCount = subjCount + 1;
end

cd(oldFolder)
%% reorder edf for following processes
edfData = edf.'; edfData = edfData(:);
runs     = repmat(1:numel(run), 1, numel(subject));
subjects = repmat(1:numel(subject), numel(run), 1);

%% find the fixation center of the data
% predefine veriables
fullGazeData = cell(numel(edf), 1);
screenWidth  = cell(numel(edf), 1);
screenHeight = cell(numel(edf), 1);
center       = cell(numel(edf), 1);
for currSamp = 1:numel(edf)
    %% loop variables
    currSubj   = subject(subjects(currSamp));
    currRun    = run(runs(currSamp));
    currData   = edfData{currSamp};
    
    %% find the screen coordination data
    infoName = 'GAZE_COORDS';
    screenSize = cellfun(@(x) strncmp(x, infoName, numel(infoName)), currData.Events.Messages.info(:));
    screenSize = strsplit(currData.Events.Messages.info{screenSize}, " ");
    screenSize = str2double(screenSize(end-3:end));

    screenWidth{currSamp}  = screenSize(3) + 1 - screenSize(1);
    screenHeight{currSamp} = screenSize(4) + 1 - screenSize(2);
    currWidth  = screenWidth{currSamp};
    currHeight = screenHeight{currSamp};
    
    %% find the recording configuration (Hz)
    infoName = 'RECCFG';
    recordingConfig = cellfun(@(x) strncmp(x, infoName, numel(infoName)), currData.Events.Messages.info(:));
    expression = '\d{3,4}'; % RECCFG CR 500 2 1 R --> e.g. expression for 500Hz
    matchStr = regexp(currData.Events.Messages.info{recordingConfig}, expression, 'match');
    frequency = str2double(matchStr{1});
    
    %% split data for each trial
    triggerName = 'stimulus Onset';
    
    % get timestamps of the trigger
    triggerTimestamps = currData.Events.Messages.time(cellfun(@(x) strncmp(x, triggerName, numel(triggerName)), currData.Events.Messages.info(:)));

    % create array from one trigger to the next (end for the last trigger is the end of the timeline)
    triggerIdx = arrayfun(@(x) find(x == currData.Samples.time), triggerTimestamps.', 'UniformOutput', false);
    triggerDataRange = [triggerTimestamps; triggerTimestamps(2:end) - 1, max(currData.Samples.time)];
   
    %% run for each trial (range from 'stimuli presentation'
    allIndices = 1:numel(currData.Samples.time);
    for currentRange = 1:size(triggerDataRange, 2)
        % Due to it's not sure if the exact timestamp of the trigger is available in the recordings, the indices are
        % selected as range
        rangeIndices = allIndices(currData.Samples.time >= triggerDataRange(1, currentRange) ...
                                & currData.Samples.time < triggerDataRange(2, currentRange));
        
        triggerInRangeBool = currData.Events.Messages.time >= triggerDataRange(1, currentRange) ...
                           & currData.Events.Messages.time < triggerDataRange(2, currentRange);
        
        % Due to trigger timestamp does not match with frequency, adjust it
        triggerTimeing = round((currData.Events.Messages.time(triggerInRangeBool) - triggerDataRange(1, currentRange))./(1000/frequency)).'; 
        trigger = table(triggerTimeing, string(currData.Events.Messages.info(triggerInRangeBool)).', 'VariableNames', {'time', 'info'});
                       
        figure(currSamp * 100 + currRun * 10 + currentRange); clf;
        %% plot gaze data
        subplot(2, 2, 1);

        posX = currData.Samples.posX(rangeIndices);
        % Y must be inverted, because eyetracker origin
        % is upper left corner in a graph its the lower left
        posY = currData.Samples.posY(rangeIndices) * -1;
        plot(posX, posY, 'o', 'Color','blue'); 
    
        title('Plot of the eye movement');
        axis([min(posX) - 1 max(posX) + 1 min(posY) - 1 max(posY) + 1]);
        axis('square');
        xlabel('x-Position');
        ylabel('y-Position');
        
        %% plot heatmap
        subplot(2, 2, 2);
        [heatmap, ~, axisRange] = currData.heatmap(min(rangeIndices), max(rangeIndices));
        imhandle = imagesc(heatmap);

        set(imhandle.Parent, 'YDir','normal');
        axis(axisRange);
        axis square;
        colorbar;
        title('HeatMap of the eye movement');
        xlabel('x-Position (shifted zero)');
        ylabel('y-Position (shifted zero)');

        
        %% plot pupil diameter
        subplot(2, 2, 3);
        plot(currData.Samples.pupilSize(rangeIndices), 'Color', 'blue')
        hold on
        lineXpos = repmat(trigger.time, 1, 2);
        y = ylim;
        lineYpos = repmat([y(1), y(2)], height(trigger), 1);
        line(lineXpos.', lineYpos.', 'Color', 'red')
        hold off
        title('Pupil Size and triggers');
        xlabel('data point (ca. ms)');
        if logical(currData.PUPIL.AREA)
            ylabel('Area (SR Research points)');
        else
            ylabel('Diameter (SR Research points)')
        end
        
        %% add trigger names to last subplot
        subplot(2, 2, 4);
        test = rowfun(@(x, y) string(sprintf('%d: %s\n', x, y)), trigger, 'OutputVariableNames', 'description');
        text(0,0.5, join(test.description)); axis off
        
    end
end
