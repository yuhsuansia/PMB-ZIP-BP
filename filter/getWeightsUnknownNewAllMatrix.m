function [weights,updatedExistence] = getWeightsUnknownNewAllMatrix(logWeights,oldExistence,numSkipColumns)
%GETWEIGHTSUNKNOWNNEWALLMATRIX Compute all extrinsic particle weights at once.
% Each output column excludes the matching measurement column from the
% summed log-weights, which is exactly the extrinsic quantity needed by BP.

if nargin < 3 || isempty(numSkipColumns)
    numSkipColumns = size(logWeights,2) - 1;
end

numSkipColumns = max(numSkipColumns,0);
numParticles = size(logWeights,1);

if numSkipColumns == 0
    weights = zeros(numParticles,0);
    updatedExistence = zeros(1,0);
    return;
end

logSumWeights = sum(logWeights,2);
extrinsicLogWeights = bsxfun(@minus, logSumWeights, logWeights(:,1:numSkipColumns));

[log_w,log_sum_w] = normalizeLogWeightsMatrix(extrinsicLogWeights);

weights = exp(log_w);
updatedExistence = zeros(1,numSkipColumns);

positiveInf = isinf(log_sum_w) & (log_sum_w > 0);
negativeInf = isinf(log_sum_w) & (log_sum_w < 0);
updatedExistence(positiveInf) = double(oldExistence > 0);
updatedExistence(negativeInf) = 0;

finiteMask = ~(positiveInf | negativeInf);
if any(finiteMask)
    aliveProbability = oldExistence * exp(log_sum_w(finiteMask));
    denominator = aliveProbability + 1 - oldExistence;
    existenceValues = aliveProbability ./ denominator;

    invalidDenominator = denominator <= 0;
    if any(invalidDenominator)
        existenceValues(invalidDenominator) = double(aliveProbability(invalidDenominator) > 0);
    end

    updatedExistence(finiteMask) = existenceValues;
end

updatedExistence = min(max(updatedExistence,0),1);
end
