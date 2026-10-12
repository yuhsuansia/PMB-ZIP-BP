function result = runDemo(model, options)
%RUNDEMO Shared setup for a single reproducible filter execution.

defaults = struct('numSteps', 100, 'numParticles', 1000, ...
    'numOuterIterations', 3, 'measurementRate', 5, ...
    'detectionProbability', 0.8, 'meanClutter', 10, 'seed', 1, 'makePlots', true);
validateattributes(options, {'struct'}, {'scalar'});
names = fieldnames(options);
for index = 1:numel(names)
    name = names{index};
    if ~isfield(defaults, name)
        error('pmbbp:UnknownOption', 'Unknown demo option: %s.', name);
    end
    defaults.(name) = options.(name);
end
options = defaults;
for name = {'numSteps', 'numParticles', 'numOuterIterations'}
    validateattributes(options.(name{1}), {'numeric'}, ...
        {'scalar', 'integer', 'positive', 'finite'}, mfilename, name{1});
end
validateattributes(options.measurementRate, {'numeric'}, {'scalar', 'positive', 'finite'});
validateattributes(options.meanClutter, {'numeric'}, {'scalar', 'positive', 'finite'});
validateattributes(options.detectionProbability, {'numeric'}, ...
    {'scalar', '>=', 0, '<=', 1, 'finite'});
validateattributes(options.seed, {'numeric'}, ...
    {'scalar', 'integer', '>=', 0, '<=', 2^32-1, 'finite'});
validateattributes(options.makePlots, {'logical'}, {'scalar'});

% Keep the caller's random stream intact while making this run reproducible.
previousRng = rng;
rngScope = onCleanup(@() rng(previousRng)); %#ok<NASGU>
parameters = buildParameters(options.numSteps, options.numParticles, options.numOuterIterations);
parameters.meanMeasurements = options.measurementRate;
parameters.priorGammaAlpha = options.measurementRate * parameters.priorGammaBeta;
parameters.birthPriorGammaAlpha = options.measurementRate * parameters.birthPriorGammaBeta;
parameters.meanClutter = options.meanClutter;
if strcmp(model, 'poisson')
    parameters.detectionProbability = 1;
else
    parameters.detectionProbability = options.detectionProbability;
end
parameters = matchPredictionParameters(parameters);

% Ten objects approach the center from a circle. Lifetimes are scaled to
% the requested scan count, so short examples retain births and deaths.
birthFractions = repelem([0.03, 0.06, 0.09, 0.12, 0.15], 2);
appearance = [max(1, round(birthFractions * options.numSteps)); ...
    max(1, round((birthFractions + 0.80) * options.numSteps))];
cfg = struct('numObjects', 10, 'numSteps', options.numSteps, ...
    'startRadius', 125, 'startSpeed', 12.5, 'appearance', appearance);
if strcmp(model, 'poisson')
    data = generateScenario(cfg, parameters, options.seed);
    filter = @eotEllipticalShapeRFS;
else
    data = generateScenarioZIP(cfg, parameters, options.seed);
    filter = @eotEllipticalShapeRFS_ZIP;
end
rng(options.seed, 'twister');
[estimates, runTime, posteriors, elapsedTime] = filter(data.measCell, parameters);
result = struct('model', model, 'options', options, 'parameters', parameters, ...
    'data', data, 'estimates', {estimates}, 'pmbPosteriors', {posteriors}, ...
    'runTime', runTime, 'elapsedTime', elapsedTime);
fprintf('PMB-BP (%s): %d scans, %d particles, %.3f s\n', ...
    model, options.numSteps, options.numParticles, runTime);
if options.makePlots
    plotDemo(result);
end
end
