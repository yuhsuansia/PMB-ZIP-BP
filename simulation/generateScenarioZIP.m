function data = generateScenarioZIP(cfg, par, seed)
%GENERATESCENARIOZIP Generate truth and ZIP measurements for one run.

parScenario = par;
if isfield(cfg, 'extentNominal') && ~isempty(cfg.extentNominal)
    parScenario.priorExtent1 = cfg.extentNominal * (parScenario.priorExtent2 - 3);
end

if isfield(parScenario, 'truthSeed') && ~isempty(parScenario.truthSeed)
    rng(parScenario.truthSeed);
else
    rng default
end
[startStates, startExtents, startRates] = ...
    getStartStates(cfg.numObjects, cfg.startRadius, cfg.startSpeed, parScenario);

[truthTracks, truthExtents, truthRates] = ...
    generateTracksUnknown(parScenario, startStates, startExtents, startRates, cfg.appearance, cfg.numSteps);

if nargin < 3 || isempty(seed)
    seed = 0;
end
rng(seed)
measCell = generateClutteredMeasurementsZIP(truthTracks, truthExtents, truthRates, parScenario);

data = struct();
data.startStates = startStates;
data.startExtents = startExtents;
data.startRates = startRates;

data.truthTracks = truthTracks;
data.truthExtents = truthExtents;
data.truthRates = truthRates;

data.measCell = measCell;
data.appearance = cfg.appearance;
data.numObjects = cfg.numObjects;
data.numSteps = cfg.numSteps;
end
