function experimentInfo = addInfo(varargin)
    % addInfo: creates the experimental information for the whole experiment,
    % inclusively the default parameters. If some variables should be changed,
    % add them as key, value pair.
    %
    % SYNTAX:   experimentInfo = ...
    %               experiment.addInfo('nrTrials',   nrTrials, ...
    %                                  'screenRect', screenRect, ...
    %                                  'ex pToolbox', expToolbox, ...
    %                                  'fontSize',   fontSize, ...
    %                                  'fontName',   fontName, ...
    %                                  'allowedOffset', allowedOffset)
    %
    % INPUTS e.g.:
    %           nrTrials       	number of Trials of the individual tasks
    %           expToolbox      used toolbox for visualization
    %                           (default: 'psychtoolbox')
    %           fontSize        default font size for the experiment
    %                           (default: 65)
    %           fontName        default font name for the experiment
    %                           (default: 'Arial')
    %           allowedOffset   allowed offset for the whole experiment timeline
    %                           (default: 0.001)
    %
    %           Further options: title, experimenter, nrRuns,
    %           screenNumber, goFreq, noGoFreq
    %
    % OUTPUTS:
    %           experimentInfo  structure of the experiment information
    %                           inclusively default parameters
    %
    % EXAMPLE:
    %           nrTrials  = 3;
    %           taskOrder = readtable('taskOrder.txt');
    %           trials    = experiment.createTasks(taskOrder, nrTrials);
    %           stages    = experiment.randomize(trials);
    %           stages    = experiment.calculateAbsoluteTimeline(stages);
    %           result    = experiment.run(stages);
    %
    % Other Classes required:
    %           Working Psychtoolbox installation

    
    
    experimentInfo = struct( ...
        'title',              '', ...
        'experimenter',       '', ...
        'participantID',            '', ...
        'participantAge',      '', ...
        'participantGender',      '', ...
        'numTimesScanned',      '', ...
        'experimentDate',               nan, ...
        'expToolbox',         '', ...
        'nrRuns',             nan, ...
        'nrTrials',           nan, ...
        'audioPair',        nan, ...
        'tactilePair',        nan, ...
        'modality0AFrequency',             nan, ...
        'modality0BFrequency',           nan, ...
        'modality0CFrequency',             nan, ...
        'modality1AFrequency',           nan, ...
        'modality1BFrequency',             nan, ...
        'modality1CFrequency',           nan, ...
        'modality2AFrequency',             nan, ...
        'modality2BFrequency',           nan, ...
        'modality2CFrequency',           nan, ...
        'responseTimeThreshold',     nan ...
        );

    
    % default values
    defaultTitle         = 'Default Title';
    defaultExperimenter  = 'Unknown';
    defaultparticipantID       = 'Unknown';
    defaultparticipantAge       = 'Unknown';
    defaultToolbox       = 'psychtoolbox';
    defaultnrRuns     = 5;
    defaultnrTrials      = 12;

    % initialize input parser
    inPars = inputParser;    
    
    % optional parameter inputs
    inPars.addParameter('title', defaultTitle, @ischar);
    inPars.addParameter('experimenter', defaultExperimenter, @ischar);
    inPars.addParameter('participantID',      defaultparticipantID,      @ischar);
    inPars.addParameter('participantAge',      defaultparticipantAge,      @ischar);
    inPars.addParameter('expToolbox',   defaultToolbox,      @ischar);
    inPars.addParameter('nrRuns',       defaultnrRuns,       @(x) isnumeric(x) && all(x) > 0);
    inPars.addParameter('nrTrials',     defaultnrTrials,     @(x) isnumeric(x) && all(x) > 0);
    inPars.addParameter('audioPair',     @(x) isnumeric(x));
    inPars.addParameter('tactilePair',     @(x) isnumeric(x));
    inPars.addParameter('modality0AFrequency',     @(x) isnumeric(x));
    inPars.addParameter('modality0BFrequency',     @(x) isnumeric(x));
    inPars.addParameter('modality0CFrequency',     @(x) isnumeric(x));
    inPars.addParameter('modality1AFrequency',    @(x) isnumeric(x));
    inPars.addParameter('modality1BFrequency',    @(x) isnumeric(x));
    inPars.addParameter('modality1CFrequency',    @(x) isnumeric(x));
    inPars.addParameter('modality2AFrequency',    @(x) isnumeric(x));
    inPars.addParameter('modality2BFrequency',    @(x) isnumeric(x));
    inPars.addParameter('modality2CFrequency',    @(x) isnumeric(x));
    inPars.addParameter('responseTimeThreshold',    @(x) isnumeric(x));
    
    inPars.parse(varargin{:});
    
    % fill parsed input to experimentInfo
    experimentInfo.title           = inPars.Results.title;
    experimentInfo.experimenter    = inPars.Results.experimenter;
    experimentInfo.participantID         = inPars.Results.participantID;
    experimentInfo.participantAge         = inPars.Results.participantAge;
    experimentInfo.expToolbox      = inPars.Results.expToolbox;
    experimentInfo.nrRuns          = inPars.Results.nrRuns;
    experimentInfo.nrTrials        = inPars.Results.nrTrials;

    
end
