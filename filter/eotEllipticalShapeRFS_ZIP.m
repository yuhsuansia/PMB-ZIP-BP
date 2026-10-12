function [estimates, runTime, pmbPosteriors, elapsedTime] = eotEllipticalShapeRFS_ZIP(measurementsCell, parameters)
%EOTELLIPTICALSHAPERFS_ZIP PMB-BP filter with zero-inflated Poisson measurements.
% measurementsCell: one 2-by-M array per scan.
% Outputs: estimates, total time, PMB moment summaries, and per-scan time.

measurementsCell = measurementsCell(:);

numParticles = parameters.numParticles;
numNewBornParticles = parameters.numNewBornParticles;

meanClutter = parameters.meanClutter;
meanBirths = parameters.meanBirths;
survivalProbability = parameters.survivalProbability;
detectionProbability = getKnownDetectionProbability(parameters);

priorVelocityCovariance = parameters.priorVelocityCovariance;
surveillanceRegion = parameters.surveillanceRegion;
areaSize =  (surveillanceRegion(2,1)-surveillanceRegion(1,1)) * ...
    (surveillanceRegion(2,2)-surveillanceRegion(1,2));
clutterIntensity = meanClutter / areaSize;
measurementsCovariance = parameters.measurementVariance * eye(2);
if isfield(parameters,'birthPriorExtent2') && isfield(parameters,'birthPriorExtent1')
    birthPriorExtent2 = parameters.birthPriorExtent2;
    birthPriorExtent1 = parameters.birthPriorExtent1;
else
    birthPriorExtent2 = parameters.priorExtent2;
    birthPriorExtent1 = parameters.priorExtent1;
end

meanExtentPrior = birthPriorExtent1/(birthPriorExtent2-3);
totalCovariance = meanExtentPrior+measurementsCovariance;
if isfield(parameters,'birthPriorGammaBeta') && isfield(parameters,'birthPriorGammaAlpha')
    birthPriorGammaBeta = parameters.birthPriorGammaBeta;
    birthPriorGammaAlpha = parameters.birthPriorGammaAlpha;
else
    birthPriorGammaBeta = parameters.priorGammaBeta;
    birthPriorGammaAlpha = parameters.priorGammaAlpha;
end
proposalCovariance = 2 * totalCovariance;
proposalCovarianceChol = chol(proposalCovariance,'lower');
priorVelocityCovarianceChol = chol(priorVelocityCovariance,'lower');

detectionThreshold = parameters.detectionThreshold;
thresholdPruning = parameters.thresholdPruning;
numOuterIterations = parameters.numOuterIterations;

[numSteps, ~] = size(measurementsCell);
probabilityPPPDetectionNoMeasurement = detectionProbability * ...
    (birthPriorGammaBeta / (birthPriorGammaBeta + 1))^birthPriorGammaAlpha;
probabilityPPPMissedDetect = 1 - detectionProbability + probabilityPPPDetectionNoMeasurement;
uniformWeight = log(1/areaSize);
bernoulliTemplate = struct( ...
    'mark', [], ...
    'existence', [], ...
    'particleWeights', [], ...
    'particlesKinematic', [], ...
    'particlesExtent', [], ...
    'particlesRate', []);
birthContext = struct( ...
    'numParticles', numNewBornParticles, ...
    'proposalCovariance', proposalCovariance, ...
    'proposalCovarianceChol', proposalCovarianceChol, ...
    'priorVelocityCovarianceChol', priorVelocityCovarianceChol, ...
    'measurementsCovariance', measurementsCovariance, ...
    'birthPriorExtent1', birthPriorExtent1, ...
    'birthPriorExtent2', birthPriorExtent2, ...
    'birthPriorGammaAlpha', birthPriorGammaAlpha, ...
    'birthPriorGammaBeta', birthPriorGammaBeta, ...
    'detectionProbability', detectionProbability, ...
    'uniformWeight', uniformWeight, ...
    'bernoulliTemplate', bernoulliTemplate);

multiBernoulli = repmat(bernoulliTemplate,[0,1]);
meanUndetectedObjects = 0;

estimates = cell(numSteps,1);
elapsedTime = zeros(numSteps,1);
if nargout > 2
    pmbPosteriors = cell(numSteps,1);
end

for step = 1:numSteps
    stepTimer = tic;
    estimates{step} = struct('mean',{},'extentMean',{});

    measurements = measurementsCell{step};
    numMeasurements = size(measurements,2);

    % Predict the legacy Bernoullis and the undetected PPP intensity.
    multiBernoulli = performBernoulliPrediction(multiBernoulli, parameters);
    numLegacy = numel(multiBernoulli);
    baseLogWeightsExisting = getBernoulliBaseLogWeights(multiBernoulli);
    meanUndetectedObjects = survivalProbability * meanUndetectedObjects + meanBirths;

    % Reorder the measurements and compute the legacy single-measurement
    % log-likelihoods used by the ZIP BP update.
    [newIndexes, measurements, logLikelihoodExisting] = prepareMeasurementsZIP( ...
        multiBernoulli, measurements, parameters);

    % Candidate new Bernoullis follow the same sparse measurement-driven
    % construction as the existing PPP/Poisson implementation.
    birthContext.aliveProbability = meanUndetectedObjects;
    [newMultiBernoulli, logLikelihoodNew, supportLengthsNew] = ...
        createSparseNewBernoullisZIP(newIndexes, measurements, step, birthContext);
    numNew = numel(newMultiBernoulli);
    baseLogWeightsNew = getBernoulliBaseLogWeights(newMultiBernoulli);

    % In the full paper formulation, every measurement owns a dedicated new
    % Bernoulli whose empty-state branch carries clutter. After sparse birth
    % generation, the dropped measurements would otherwise lose that clutter
    % alternative. Preserve it as a collapsed measurement-side term.
    missingClutterMass = getMissingClutterMass(numMeasurements, supportLengthsNew, clutterIntensity);

    % Initialize the extrinsic Bernoulli messages.
    [existenceExtrinsicExisting, logWeightsExtrinsicExisting] = ...
        initializeExistingExtrinsics(multiBernoulli, baseLogWeightsExisting, ...
        numMeasurements, detectionProbability);
    [existenceExtrinsicNew, logWeightsExtrinsicNew] = ...
        initializeNewExtrinsics(newMultiBernoulli, baseLogWeightsNew, supportLengthsNew);
    [chiExistenceExisting, logWeightsChiExisting] = initializeChiMessages( ...
        multiBernoulli, baseLogWeightsExisting);
    [xiExisting, xiNew] = initializeZIPAssociationMessages( ...
        numLegacy, numMeasurements, supportLengthsNew, missingClutterMass);

    % Loopy BP: update the extrinsic Bernoulli messages on fixed particle
    % supports, then form the final beliefs from the last message state.
    for outer = 1:max(numOuterIterations - 1, 0)
        messageState = buildZIPMessageState( ...
            existenceExtrinsicExisting, logWeightsExtrinsicExisting, logLikelihoodExisting, ...
            existenceExtrinsicNew, logWeightsExtrinsicNew, logLikelihoodNew, ...
            chiExistenceExisting, logWeightsChiExisting, multiBernoulli, ...
            xiExisting, xiNew, detectionProbability, clutterIntensity, missingClutterMass);
        [existenceExtrinsicExisting, logWeightsExtrinsicExisting] = ...
            updateExistingExtrinsicsZIP(multiBernoulli, baseLogWeightsExisting, messageState);
        [existenceExtrinsicNew, logWeightsExtrinsicNew] = ...
            updateNewExtrinsicsZIP(newMultiBernoulli, baseLogWeightsNew, messageState);
        [chiExistenceExisting, logWeightsChiExisting] = updateChiMessagesZIP( ...
            multiBernoulli, baseLogWeightsExisting, messageState);
        xiExisting = messageState.xiExistingNext;
        xiNew = messageState.xiNewNext;
    end
    messageState = buildZIPMessageState( ...
        existenceExtrinsicExisting, logWeightsExtrinsicExisting, logLikelihoodExisting, ...
        existenceExtrinsicNew, logWeightsExtrinsicNew, logLikelihoodNew, ...
        chiExistenceExisting, logWeightsChiExisting, multiBernoulli, ...
        xiExisting, xiNew, detectionProbability, clutterIntensity, missingClutterMass);

    % Final Bernoulli beliefs.
    for target = 1:numLegacy
        multiBernoulli(target) = updateBernoulliZIPBelief( ...
            multiBernoulli(target), messageState.logZetaSingletonExisting{target}, ...
            messageState.logZetaEmptyExisting(target), ...
            messageState.logRhoSingletonExisting{target}, ...
            messageState.logRhoEmptyExisting{target}, ...
            baseLogWeightsExisting{target});

        if multiBernoulli(target).existence > detectionThreshold
            estimates{step}(end+1) = summarizeBernoulliEstimate(multiBernoulli(target)); %#ok<AGROW>
        end
        multiBernoulli(target) = resampleBernoulliTargets(multiBernoulli(target), numParticles);
    end

    for target = 1:numNew
        targetIndex = numLegacy + target;
        newPosterior = updateBernoulliZIPBelief( ...
            newMultiBernoulli(target), 0, 0, ...
            messageState.logRhoSingletonNew{target}, ...
            messageState.logRhoEmptyNew{target}, ...
            baseLogWeightsNew{target});

        if numNewBornParticles > numParticles
            newPosterior = resampleBernoulliTargets(newPosterior, numParticles);
        end
        if newPosterior.existence > detectionThreshold
            estimates{step}(end+1) = summarizeBernoulliEstimate(newPosterior); %#ok<AGROW>
        end
        multiBernoulli(targetIndex) = newPosterior; %#ok<AGROW>
    end

    isNotRedundant = [multiBernoulli.existence] >= thresholdPruning;
    multiBernoulli = multiBernoulli(isNotRedundant);
    if nargout > 2
        pmbPosteriors{step} = summarizePMBPosterior(multiBernoulli);
    end

    meanUndetectedObjects = meanUndetectedObjects * probabilityPPPMissedDetect;
    elapsedTime(step) = toc(stepTimer);
end

runTime = sum(elapsedTime);
end

function [newMultiBernoulli, logLikelihoodNew, supportLengthsNew] = ...
        createSparseNewBernoullisZIP(newIndexes, measurements, step, birthContext)
numNew = numel(newIndexes);
newMultiBernoulli = repmat(birthContext.bernoulliTemplate,[numNew,1]);
logLikelihoodNew = cell(numNew,1);
supportLengthsNew = zeros(numNew,1);

for target = 1:numNew
    supportLengthsNew(target) = newIndexes(target);
    proposalMean = measurements(:,supportLengthsNew(target));
    newMultiBernoulli(target) = initializeNewBernoulliZIP( ...
        proposalMean, step, supportLengthsNew(target), birthContext);
    logLikelihoodNew{target} = evaluateNewBernoulliLikelihoodsZIP( ...
        newMultiBernoulli(target), measurements(:,1:supportLengthsNew(target)), birthContext);
end
end

function newBernoulli = initializeNewBernoulliZIP(proposalMean, step, supportLength, birthContext)
numBirthParticles = birthContext.numParticles;
newBernoulli = birthContext.bernoulliTemplate;
newBernoulli.particlesKinematic = zeros(4,numBirthParticles);
newBernoulli.particlesKinematic(1:2,:) = ...
    proposalMean + birthContext.proposalCovarianceChol * randn(2,numBirthParticles);
newBernoulli.particlesKinematic(3:4,:) = ...
    birthContext.priorVelocityCovarianceChol * randn(2,numBirthParticles);
newBernoulli.particlesExtent = iwishrndFastVector( ...
    birthContext.birthPriorExtent1, birthContext.birthPriorExtent2, numBirthParticles);
newBernoulli.particlesRate = gamrnd( ...
    birthContext.birthPriorGammaAlpha, 1/birthContext.birthPriorGammaBeta, ...
    numBirthParticles, 1);
newBernoulli.particleWeights = exp(normalizeLogWeights( ...
    birthContext.uniformWeight - gaussianLogPdf( ...
    reshape(newBernoulli.particlesKinematic(1:2,:),2,numBirthParticles)', ...
    proposalMean', birthContext.proposalCovariance')));
aliveProbability = birthContext.aliveProbability * birthContext.detectionProbability * ...
    (newBernoulli.particleWeights' * exp(-newBernoulli.particlesRate));
newBernoulli.existence = aliveProbability / (aliveProbability + 1);
newBernoulli.mark = [step;supportLength];
end

function logLikelihood = evaluateNewBernoulliLikelihoodsZIP(newBernoulli, measurements, birthContext)
numBirthParticles = birthContext.numParticles;
particleMeans = reshape(newBernoulli.particlesKinematic(1:2,:),2,numBirthParticles);
particleCovariances = reshape(newBernoulli.particlesExtent,2,2,numBirthParticles) + ...
    repmat(birthContext.measurementsCovariance,[1,1,numBirthParticles]);
logLikelihood = bsxfun(@plus, logZero(newBernoulli.particlesRate(:)), ...
    getLogWeightsFastMatrix(measurements, particleMeans, particleCovariances));
end

function missingClutterMass = getMissingClutterMass(numMeasurements, supportLengthsNew, clutterIntensity)
hasDirectClutterBranch = false(1,numMeasurements);
hasDirectClutterBranch(supportLengthsNew) = true;
missingClutterMass = clutterIntensity * (~hasDirectClutterBranch);
end

function [existenceExtrinsic, logWeightsExtrinsic] = initializeExistingExtrinsics( ...
        multiBernoulli, baseLogWeights, numMeasurements, detectionProbability)
numLegacy = numel(multiBernoulli);
existenceExtrinsic = zeros(numLegacy, numMeasurements);
logWeightsExtrinsic = cell(numLegacy,1);
logOneMinusDetectionProbability = logZero(1 - detectionProbability);
logDetectionProbability = logZero(detectionProbability);

for target = 1:numLegacy
    priorExistence = multiBernoulli(target).existence;
    logZipMissProbability = logSumExpPair( ...
        logOneMinusDetectionProbability * ones(size(baseLogWeights{target})), ...
        logDetectionProbability - multiBernoulli(target).particlesRate(:));
    [logWeightsCurrent, logAverageMissProbability] = normalizeLogWeights( ...
        baseLogWeights{target} + logZipMissProbability);
    averageMissProbability = exp(logAverageMissProbability);
    existenceExtrinsicCurrent = priorExistence * averageMissProbability / ...
        max(1 - priorExistence + priorExistence * averageMissProbability, eps);
    existenceExtrinsic(target,:) = existenceExtrinsicCurrent;
    logWeightsExtrinsic{target} = repmat(logWeightsCurrent, 1, numMeasurements);
end
end

function [chiExistence, logWeightsChi] = initializeChiMessages(multiBernoulli, baseLogWeights)
numLegacy = numel(multiBernoulli);
chiExistence = zeros(numLegacy,1);
logWeightsChi = cell(numLegacy,1);

for target = 1:numLegacy
    chiExistence(target) = multiBernoulli(target).existence;
    logWeightsChi{target} = baseLogWeights{target};
end
end

function [existenceExtrinsic, logWeightsExtrinsic] = initializeNewExtrinsics( ...
        newMultiBernoulli, baseLogWeights, supportLengths)
numNew = numel(newMultiBernoulli);
existenceExtrinsic = cell(numNew,1);
logWeightsExtrinsic = cell(numNew,1);

for target = 1:numNew
    existenceExtrinsic{target} = repmat(newMultiBernoulli(target).existence, 1, supportLengths(target));
    logWeightsExtrinsic{target} = repmat(baseLogWeights{target}, 1, supportLengths(target));
end
end

function [xiExisting, xiNew] = initializeZIPAssociationMessages( ...
        numLegacy, numMeasurements, supportLengthsNew, missingClutterMass)
connectedCounts = numLegacy + sum(bsxfun(@ge, supportLengthsNew(:), 1:numMeasurements), 1);
initialXi = connectedCounts - 1 + missingClutterMass;

xiExisting = repmat(initialXi, numLegacy, 1);
numNew = numel(supportLengthsNew);
xiNew = cell(numNew,1);
for target = 1:numNew
    xiNew{target} = initialXi(1:supportLengthsNew(target));
end
end

function messageState = buildZIPMessageState( ...
        existenceExtrinsicExisting, logWeightsExtrinsicExisting, logLikelihoodExisting, ...
        existenceExtrinsicNew, logWeightsExtrinsicNew, logLikelihoodNew, ...
        chiExistenceExisting, logWeightsChiExisting, multiBernoulli, xiExistingCurrent, ...
        xiNewCurrent, detectionProbability, clutterIntensity, missingClutterMass)

numLegacy = size(existenceExtrinsicExisting,1);
numMeasurements = size(existenceExtrinsicExisting,2);
numNew = numel(existenceExtrinsicNew);
logExistenceExtrinsicExisting = logZero(existenceExtrinsicExisting);
logXiExistingCurrent = logZero(xiExistingCurrent);
logOneMinusDetectionProbability = logZero(1 - detectionProbability);
logDetectionNoMeasurementProbability = logZero(detectionProbability);

messageState = struct();
messageState.logZetaSingletonExisting = cell(numLegacy,1);
messageState.logZetaEmptyExisting = zeros(numLegacy,1);
messageState.logRhoSingletonExisting = cell(numLegacy,1);
messageState.logRhoEmptyExisting = cell(numLegacy,1);
messageState.logRhoSingletonNew = cell(numNew,1);
messageState.logRhoEmptyNew = cell(numNew,1);
messageState.xiExistingNext = zeros(numLegacy, numMeasurements);
messageState.xiNewNext = cell(numNew,1);

logDetectionNoMeasurementPerParticle = cell(numLegacy,1);
for target = 1:numLegacy
    logDetectionNoMeasurementPerParticle{target} = ...
        logDetectionNoMeasurementProbability - multiBernoulli(target).particlesRate(:);
end

if numMeasurements == 0
    for target = 1:numLegacy
        numParticles = size(logWeightsExtrinsicExisting{target},1);
        messageState.logZetaSingletonExisting{target} = logSumExpPair( ...
            logOneMinusDetectionProbability * ones(numParticles,1), ...
            logDetectionNoMeasurementPerParticle{target});
        messageState.logRhoSingletonExisting{target} = zeros(numParticles,0);
        messageState.logRhoEmptyExisting{target} = zeros(1,0);
    end
    return;
end

thetaExisting = zeros(numLegacy, numMeasurements);
for target = 1:numLegacy
    logTheta = logExistenceExtrinsicExisting(target,:) + ...
        logSumExpColumns(bsxfun(@plus, ...
            logWeightsExtrinsicExisting{target}, ...
            logLikelihoodExisting{target}));
    thetaExisting(target,:) = exp(logTheta);
    thetaExisting(target,~isfinite(thetaExisting(target,:))) = 0;
end

mu1 = xiExistingCurrent + thetaExisting;
logMu1Existing = logZero(mu1);
logProdMu0Exclude = excludeLogSumsRows(logXiExistingCurrent);
logProdMu1Exclude = excludeLogSumsRows(logMu1Existing);
logNu0 = zeros(numLegacy, numMeasurements);
logNu1 = zeros(numLegacy, numMeasurements);
logProdMu0All = sumFiniteLogsRows(logXiExistingCurrent);
logProdMu1All = sumFiniteLogsRows(logMu1Existing);
for target = 1:numLegacy
    noDetectionMass = max(1 - detectionProbability * chiExistenceExisting(target), 0);
    detectionNoMeasurementMass = exp(logZero(chiExistenceExisting(target)) + ...
        logSumExpColumns(logWeightsChiExisting{target} + ...
        logDetectionNoMeasurementPerParticle{target}));
    logNu0(target,:) = logProdMu0Exclude(target,:) + logZero(noDetectionMass);
    logNu1(target,:) = logProdMu1Exclude(target,:) + logZero(detectionNoMeasurementMass);
    messageState.logZetaSingletonExisting{target} = logSumExpPair( ...
        logOneMinusDetectionProbability + logProdMu0All(target), ...
        logDetectionNoMeasurementPerParticle{target} + logProdMu1All(target));
end
logLambdaBarExisting = logNu1 - logSumExpPair(logNu0, logNu1);
lambdaBarExisting = exp(logLambdaBarExisting);
lambdaBarExisting(~isfinite(lambdaBarExisting)) = 0;
etaExisting = thetaExisting .* lambdaBarExisting;

messageState.logZetaEmptyExisting = logProdMu0All;

etaNew = cell(numNew,1);
for target = 1:numNew
    numValidMeasurements = size(logLikelihoodNew{target},2);
    logTheta = logZero(existenceExtrinsicNew{target}) + ...
        logSumExpColumns(bsxfun(@plus, ...
            logWeightsExtrinsicNew{target}, ...
            logLikelihoodNew{target}));
    thetaCurrent = exp(logTheta);
    thetaCurrent(~isfinite(thetaCurrent)) = 0;
    if ~isempty(thetaCurrent)
        denominator = max(1 - existenceExtrinsicNew{target}(numValidMeasurements), eps);
        thetaCurrent(numValidMeasurements) = ...
            thetaCurrent(numValidMeasurements) / denominator + clutterIntensity;
    end
    etaNew{target} = thetaCurrent;
end

sumEta = sum(etaExisting, 1);
for target = 1:numNew
    numValidMeasurements = numel(etaNew{target});
    sumEta(1:numValidMeasurements) = sumEta(1:numValidMeasurements) + etaNew{target};
end
sumEta = sumEta + missingClutterMass;

messageState.xiExistingNext = max(repmat(sumEta, numLegacy, 1) - etaExisting, 0);
for target = 1:numNew
    numValidMeasurements = numel(etaNew{target});
    messageState.xiNewNext{target} = max(sumEta(1:numValidMeasurements) - etaNew{target}, 0);
end

for target = 1:numLegacy
    logTau0 = logXiExistingCurrent(target,:);
    logTau1 = logLambdaBarExisting(target,:);
    messageState.logRhoSingletonExisting{target} = logSumExpWithRow( ...
        bsxfun(@plus, logLikelihoodExisting{target}, logTau1), logTau0);
    messageState.logRhoEmptyExisting{target} = logTau0;
end

for target = 1:numNew
    logTau0 = logZero(xiNewCurrent{target});
    logRhoSingleton = logSumExpWithRow(logLikelihoodNew{target}, logTau0);
    if ~isempty(logRhoSingleton)
        forcedMeasurement = size(logRhoSingleton,2);
        logRhoSingleton(:,forcedMeasurement) = logLikelihoodNew{target}(:,forcedMeasurement);
    end
    logRhoEmpty = logTau0;
    if ~isempty(logRhoEmpty)
        logRhoEmpty(end) = logSumExpPair(logRhoEmpty(end), logZero(clutterIntensity));
    end
    messageState.logRhoSingletonNew{target} = logRhoSingleton;
    messageState.logRhoEmptyNew{target} = logRhoEmpty;
end
end

function [chiExistence, logWeightsChi] = updateChiMessagesZIP(multiBernoulli, baseLogWeights, messageState)
numLegacy = numel(multiBernoulli);
chiExistence = zeros(numLegacy,1);
logWeightsChi = cell(numLegacy,1);

for target = 1:numLegacy
    singletonLogFactors = zeros(size(baseLogWeights{target}));
    if ~isempty(messageState.logRhoSingletonExisting{target})
        singletonLogFactors = sum(messageState.logRhoSingletonExisting{target}, 2);
    end

    emptyLogFactor = 0;
    if ~isempty(messageState.logRhoEmptyExisting{target})
        emptyLogFactor = sum(messageState.logRhoEmptyExisting{target});
    end

    [chiExistence(target), logWeightsChi{target}] = normalizeBernoulliFromFactorsLog( ...
        multiBernoulli(target).existence, baseLogWeights{target}, ...
        singletonLogFactors, emptyLogFactor);
end
end

function [existenceExtrinsic, logWeightsExtrinsic] = updateExistingExtrinsicsZIP( ...
        multiBernoulli, baseLogWeights, messageState)
numLegacy = numel(multiBernoulli);
numMeasurements = size(messageState.xiExistingNext,2);

existenceExtrinsic = zeros(numLegacy, numMeasurements);
logWeightsExtrinsic = cell(numLegacy,1);
for target = 1:numLegacy
    baseExistence = multiBernoulli(target).existence;
    logRhoSingleton = messageState.logRhoSingletonExisting{target};
    logRhoEmpty = messageState.logRhoEmptyExisting{target};
    logRhoExclude = excludeLogSumsColumns(logRhoSingleton);
    logRhoEmptyExclude = excludeLogSumsRow(logRhoEmpty);
    logSingletonFactors = bsxfun(@plus, logRhoExclude, ...
        messageState.logZetaSingletonExisting{target});
    logEmptyFactors = messageState.logZetaEmptyExisting(target) + logRhoEmptyExclude;
    [existenceExtrinsic(target,:), logWeightsExtrinsic{target}] = ...
        computeBernoulliExtrinsicsLog(baseExistence, baseLogWeights{target}, ...
        logSingletonFactors, logEmptyFactors);
end
end

function [existenceExtrinsic, logWeightsExtrinsic] = updateNewExtrinsicsZIP( ...
        newMultiBernoulli, baseLogWeights, messageState)
numNew = numel(newMultiBernoulli);
existenceExtrinsic = cell(numNew,1);
logWeightsExtrinsic = cell(numNew,1);

for target = 1:numNew
    baseExistence = newMultiBernoulli(target).existence;
    logRhoSingleton = messageState.logRhoSingletonNew{target};
    logRhoEmpty = messageState.logRhoEmptyNew{target};
    logRhoExclude = excludeLogSumsColumns(logRhoSingleton);
    logRhoEmptyExclude = excludeLogSumsRow(logRhoEmpty);
    [existenceExtrinsic{target}, logWeightsExtrinsic{target}] = ...
        computeBernoulliExtrinsicsLog(baseExistence, baseLogWeights{target}, ...
        logRhoExclude, logRhoEmptyExclude);
end
end

function Bernoulli = updateBernoulliZIPBelief( ...
        Bernoulli, logZetaSingleton, logZetaEmpty, logRhoSingleton, logRhoEmpty, baseLogWeights)
singletonLogFactors = logZetaSingleton;
if ~isempty(logRhoSingleton)
    singletonLogFactors = singletonLogFactors + sum(logRhoSingleton, 2);
end

emptyLogFactor = logZetaEmpty;
if ~isempty(logRhoEmpty)
    emptyLogFactor = emptyLogFactor + sum(logRhoEmpty);
end

[Bernoulli.existence, logParticleWeights] = normalizeBernoulliFromFactorsLog( ...
    Bernoulli.existence, baseLogWeights, singletonLogFactors, emptyLogFactor);
Bernoulli.particleWeights = exp(logParticleWeights);
end

function baseLogWeights = getBernoulliBaseLogWeights(BernoulliArray)
numBernoulli = numel(BernoulliArray);
baseLogWeights = cell(numBernoulli,1);
for index = 1:numBernoulli
    baseLogWeights{index} = logZero(BernoulliArray(index).particleWeights(:));
end
end

function [existenceExtrinsic, logWeightsExtrinsic] = computeBernoulliExtrinsicsLog( ...
        baseExistence, baseLogWeights, logSingletonFactors, logEmptyFactors)
numParticles = numel(baseLogWeights);
numMessages = size(logSingletonFactors,2);

existenceExtrinsic = zeros(1, numMessages);
logWeightsExtrinsic = -inf(numParticles, numMessages);
if numMessages == 0
    return;
end

logSingletonWeights = bsxfun(@plus, baseLogWeights, logSingletonFactors);
logSingletonFactorMass = logSumExpColumns(logSingletonWeights);
logWeightsExtrinsic = bsxfun(@minus, logSingletonWeights, logSingletonFactorMass);

invalidSingleton = ~isfinite(logSingletonFactorMass);
if any(invalidSingleton)
    logWeightsExtrinsic(:,invalidSingleton) = -log(max(numParticles,1));
end

logSingletonMass = logZero(baseExistence) + logSingletonFactorMass;
logEmptyMass = logZero(1 - baseExistence) + logEmptyFactors;
logTotalMass = logSumExpPair(logSingletonMass, logEmptyMass);

validMass = isfinite(logTotalMass) & ~invalidSingleton;
existenceExtrinsic(validMass) = exp(logSingletonMass(validMass) - logTotalMass(validMass));
existenceExtrinsic = min(max(existenceExtrinsic,0),1);
end

function [existence, logParticleWeights] = normalizeBernoulliFromFactorsLog( ...
        baseExistence, baseLogWeights, logSingletonFactors, logEmptyFactor)
singletonLogWeights = baseLogWeights + logSingletonFactors;
[logParticleWeights, logSingletonFactorMass] = normalizeLogWeights(singletonLogWeights);

numParticles = numel(baseLogWeights);
if isinf(logSingletonFactorMass) && logSingletonFactorMass < 0
    logParticleWeights = -log(max(numParticles,1)) * ones(numParticles,1);
    existence = 0;
    return;
end

logSingletonMass = logZero(baseExistence) + logSingletonFactorMass;
logEmptyMass = logZero(1 - baseExistence) + logEmptyFactor;
logTotalMass = logSumExpPair(logSingletonMass, logEmptyMass);

if isinf(logTotalMass) && logTotalMass < 0
    existence = 0;
    return;
end

existence = exp(logSingletonMass - logTotalMass);
existence = min(max(existence,0),1);
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

function logSumsExclude = excludeLogSumsColumns(logMatrix)
numColumns = size(logMatrix,2);
numRows = size(logMatrix,1);
logSumsExclude = -inf(numRows, numColumns);
if numColumns == 0
    return;
end

finiteMask = isfinite(logMatrix);
finiteTerms = logMatrix;
finiteTerms(~finiteMask) = 0;
finiteSums = sum(finiteTerms, 2);
numNegativeInf = sum(~finiteMask, 2);

maskFiniteRows = numNegativeInf == 0;
if any(maskFiniteRows)
    logSumsExclude(maskFiniteRows,:) = bsxfun(@minus, ...
        finiteSums(maskFiniteRows), logMatrix(maskFiniteRows,:));
end

maskSingleInfRows = numNegativeInf == 1;
if any(maskSingleInfRows)
    singleInfMask = ~finiteMask(maskSingleInfRows,:);
    singleInfRows = find(maskSingleInfRows);
    [rowOffsets, columnIndices] = find(singleInfMask);
    linearIndices = sub2ind([numRows, numColumns], singleInfRows(rowOffsets), columnIndices);
    logSumsExclude(linearIndices) = finiteSums(singleInfRows(rowOffsets));
end
end

function logSumsExclude = excludeLogSumsRow(logRow)
numColumns = numel(logRow);
logSumsExclude = -inf(1, numColumns);
if numColumns == 0
    return;
end

finiteMask = isfinite(logRow);
finiteSum = sum(logRow(finiteMask));
numNegativeInf = sum(~finiteMask);

for column = 1:numColumns
    current = logRow(column);
    if numNegativeInf == 0
        logSumsExclude(column) = finiteSum - current;
    elseif numNegativeInf == 1 && ~isfinite(current)
        logSumsExclude(column) = finiteSum;
    end
end
end

function logSumsExclude = excludeLogSumsRows(logMatrix)
numRows = size(logMatrix,1);
numColumns = size(logMatrix,2);
logSumsExclude = -inf(numRows, numColumns);
if numColumns == 0
    return;
end

finiteMask = isfinite(logMatrix);
finiteTerms = logMatrix;
finiteTerms(~finiteMask) = 0;
finiteSums = sum(finiteTerms, 2);
numNegativeInf = sum(~finiteMask, 2);

maskFiniteRows = numNegativeInf == 0;
if any(maskFiniteRows)
    logSumsExclude(maskFiniteRows,:) = bsxfun(@minus, ...
        finiteSums(maskFiniteRows), logMatrix(maskFiniteRows,:));
end

maskSingleInfRows = numNegativeInf == 1;
if any(maskSingleInfRows)
    singleInfMask = ~finiteMask(maskSingleInfRows,:);
    singleInfRows = find(maskSingleInfRows);
    [rowOffsets, columnIndices] = find(singleInfMask);
    linearIndices = sub2ind([numRows, numColumns], singleInfRows(rowOffsets), columnIndices);
    logSumsExclude(linearIndices) = finiteSums(singleInfRows(rowOffsets));
end
end

function totals = sumFiniteLogsRows(logMatrix)
finiteMask = isfinite(logMatrix);
finiteTerms = logMatrix;
finiteTerms(~finiteMask) = 0;
totals = sum(finiteTerms, 2);
totals(any(~finiteMask, 2)) = -inf;
end

function logValues = logZero(values)
logValues = -inf(size(values));
positiveMask = values > 0;
logValues(positiveMask) = log(values(positiveMask));
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

function out = logSumExpWithRow(logMatrix, logRow)
if isempty(logMatrix)
    out = logMatrix;
    return;
end

maxValue = bsxfun(@max, logMatrix, logRow);
minValue = bsxfun(@min, logMatrix, logRow);
out = maxValue;
finiteMask = isfinite(maxValue);
if any(finiteMask(:))
    out(finiteMask) = maxValue(finiteMask) + ...
        log1p(exp(minValue(finiteMask) - maxValue(finiteMask)));
end
end
