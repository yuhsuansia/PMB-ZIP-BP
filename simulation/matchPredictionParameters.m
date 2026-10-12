function parameters = matchPredictionParameters(parameters, gammaEta, iwTau)
%MATCHPREDICTIONPARAMETERS Calibrate particle prediction to Gamma/IW discounting.

if nargin < 2 || isempty(gammaEta)
    gammaEta = 1.01;
end
if nargin < 3 || isempty(iwTau)
    iwTau = parameters.scanTime * 100;
end

if gammaEta <= 1
    error('gammaEta must be larger than 1 to inflate rate uncertainty.');
end

dt = parameters.scanTime;
ff = exp(-dt / iwTau);

vRef = parameters.priorExtent2;
if vRef <= 5
    error('priorExtent2 must be larger than 5 to match extent uncertainty.');
end
vPred = 3 + ff * (vRef - 3);

% Match the one-step change in diagonal relative variance for the extent.
relVarOld = 2 / (vRef - 5);
relVarNew = 2 / (vPred - 5);
parameters.degreeFreedomPrediction1 = 2 / (relVarNew - relVarOld);

% Match the one-step variance inflation of the Gamma rate.
parameters.degreeFreedomPrediction2 = parameters.priorGammaBeta / (gammaEta - 1);
end
