function detectionProbability = getKnownDetectionProbability(parameters)
%GETKNOWNDETECTIONPROBABILITY Resolve the fixed object detection probability.

if ~isfield(parameters,'detectionProbability') || isempty(parameters.detectionProbability)
    error('parameters.detectionProbability must be defined.');
end
detectionProbability = parameters.detectionProbability;

if ~isscalar(detectionProbability) || ~isfinite(detectionProbability) || ...
        detectionProbability < 0 || detectionProbability > 1
    error('detectionProbability must be a finite scalar in [0,1].');
end
end
