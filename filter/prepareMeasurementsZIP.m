function [newIndexes, measurements, logLikelihoodExisting] = prepareMeasurementsZIP(multiBernoulli, measurements, parameters)
%PREPAREMEASUREMENTSZIP Reorder measurements and score legacy Bernoullis for ZIP BP.

numMeasurements = size(measurements,2);
numTargets = numel(multiBernoulli);

if numMeasurements == 0
    newIndexes = zeros(0,1);
    logLikelihoodExisting = cell(numTargets,1);
    return;
end

measurementsCovariance = parameters.measurementVariance * eye(2);
detectionProbability = getKnownDetectionProbability(parameters);
meanClutter = parameters.meanClutter;
surveillanceRegion = parameters.surveillanceRegion;
areaSize =  (surveillanceRegion(2,1)-surveillanceRegion(1,1)) * ...
    (surveillanceRegion(2,2)-surveillanceRegion(1,2));
logBirthScoringScaleBase = log(areaSize / meanClutter);
logOneMinusDetectionProbability = logZero(1 - detectionProbability);
logDetectionProbability = logZero(detectionProbability);

priorExistence = [multiBernoulli.existence]';
inputDA = zeros(numTargets,numMeasurements);
logLikelihoodExisting = cell(numTargets,1);

for target = 1:numTargets
    numParticles = numel(multiBernoulli(target).particleWeights);
    tmpMean = reshape(multiBernoulli(target).particlesKinematic(1:2,:),2,numParticles);
    tmpCov = reshape(multiBernoulli(target).particlesExtent,2,2,numParticles) + ...
        repmat(measurementsCovariance,[1,1,numParticles]);
    logRate = logZero(multiBernoulli(target).particlesRate(:));
    logLikelihood = getLogWeightsFastMatrix(measurements,tmpMean,tmpCov);
    logLikelihoodExisting{target} = bsxfun(@plus, ...
        logBirthScoringScaleBase + logRate, logLikelihood);

    logZipMissProbability = logSumExpPair( ...
        logOneMinusDetectionProbability * ones(numParticles,1), ...
        logDetectionProbability - multiBernoulli(target).particlesRate(:));
    [logWeightsCurrent, logAverageMissProbability] = normalizeLogWeights( ...
        logZero(multiBernoulli(target).particleWeights(:)) + logZipMissProbability);

    averageMissProbability = exp(logAverageMissProbability);
    existenceCurrent = priorExistence(target) * averageMissProbability / ...
        max(1 - priorExistence(target) + priorExistence(target) * averageMissProbability, eps);
    inputDA(target,:) = existenceCurrent .* exp(logSumExpColumns(bsxfun(@plus, ...
        logWeightsCurrent, logLikelihoodExisting{target})));
end

probabilitiesNew = 1 ./ (sum(inputDA,1)' + 1);
[newIndexes,indexesReordered] = getCentralReordered( ...
    measurements, probabilitiesNew, measurementsCovariance, parameters);
measurements = measurements(:,indexesReordered);
logLikelihoodExisting = cellfun(@(x) x(:,indexesReordered), logLikelihoodExisting, ...
    'UniformOutput', false);
end

function logSums = logSumExpColumns(logMatrix)
numColumns = size(logMatrix, 2);
logSums = -inf(1, numColumns);
if numColumns == 0
    return;
end

columnMax = max(logMatrix, [], 1);
finiteColumns = isfinite(columnMax);
if ~any(finiteColumns)
    return;
end

shifted = bsxfun(@minus, logMatrix(:,finiteColumns), columnMax(finiteColumns));
shifted(~isfinite(shifted)) = -inf;
logSums(finiteColumns) = columnMax(finiteColumns) + log(sum(exp(shifted), 1));
end

function logValues = logZero(values)
logValues = -inf(size(values));
mask = values > 0;
logValues(mask) = log(values(mask));
end

function out = logSumExpPair(a, b)
maxValue = max(a, b);
minValue = min(a, b);
out = maxValue;
finiteMask = isfinite(maxValue);
if any(finiteMask(:))
    out(finiteMask) = maxValue(finiteMask) + ...
        log1p(exp(minValue(finiteMask) - maxValue(finiteMask)));
end
end
