function [newIndexes, measurements, likelihood1] = getPromisingNewMeasurement(multiBernoulli, measurements, parameters)
%GETPROMISINGNEWMEASUREMENT Identify measurements likely to correspond to new targets.

% Score each measurement against the legacy Bernoullis only. Measurements
% with large residual "new-target" probability are reordered to the front,
% which allows the birth update to stay sparse.
numMeasurements = size(measurements,2);
numTargets = numel(multiBernoulli);
measurementsCovariance = parameters.measurementVariance * eye(2);
meanClutter = parameters.meanClutter;
surveillanceRegion = parameters.surveillanceRegion;
areaSize =  (surveillanceRegion(2,1)-surveillanceRegion(1,1)) * (surveillanceRegion(2,2)-surveillanceRegion(1,2));
currentEpsilonExtrinsic = repmat([multiBernoulli.existence]',[1,numMeasurements]);

numParticlesBernoulli = arrayfun(@(x) numel(x.particleWeights),multiBernoulli);
inputDA = zeros(numTargets,numMeasurements);

likelihood1 = cell(numTargets,1);

for target = 1:numTargets
    lengthIndex = numParticlesBernoulli(target);
    tmpMean = reshape(multiBernoulli(target).particlesKinematic(1:2,:),2,lengthIndex);
    tmpCov = reshape(multiBernoulli(target).particlesExtent,2,2,lengthIndex) + repmat(measurementsCovariance,[1,1,lengthIndex]);
    likelihoodScale = areaSize * (multiBernoulli(target).particlesRate / meanClutter);
    logLikelihood = getLogWeightsFastMatrix(measurements,tmpMean,tmpCov);
    likelihood1{target} = bsxfun(@times, likelihoodScale, exp(logLikelihood));
    inputDA(target,:) = currentEpsilonExtrinsic(target,:) .* ...
        (multiBernoulli(target).particleWeights' * likelihood1{target});
end

% The final row of the normalized DA table corresponds to the "new object"
% explanation for each measurement.
probabilitiesNew = 1 ./ (sum(inputDA,1)' + 1);

% Find cluster centers among unused measurements and reorder measurements.
[newIndexes,indexesReordered] = getCentralReordered(measurements,probabilitiesNew,measurementsCovariance,parameters);
measurements = measurements(:,indexesReordered);
likelihood1 = cellfun(@(x) x(:,indexesReordered),likelihood1,'UniformOutput',false);

end
