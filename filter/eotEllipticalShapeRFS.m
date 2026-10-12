function [estimates, runTime, pmbPosteriors, elapsedTime] = eotEllipticalShapeRFS(measurementsCell, parameters)
%EOTELLIPTICALSHAPERFS PMB-BP filter for elliptical extended-object tracking.
% measurementsCell: one 2-by-M array per scan.
% Outputs: estimates, total time, PMB moment summaries, and per-scan time.

measurementsCell = measurementsCell(:);

numParticles = parameters.numParticles;
numNewBornParticles = parameters.numNewBornParticles;

meanClutter = parameters.meanClutter;
meanBirths = parameters.meanBirths;
survivalProbability = parameters.survivalProbability;

priorVelocityCovariance = parameters.priorVelocityCovariance;
surveillanceRegion = parameters.surveillanceRegion;
areaSize =  (surveillanceRegion(2,1)-surveillanceRegion(1,1)) * (surveillanceRegion(2,2)-surveillanceRegion(1,2));
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
probabilityPPPMissedDetect = (birthPriorGammaBeta/(birthPriorGammaBeta + 1))^birthPriorGammaAlpha;
uniformWeight = log(1/areaSize);
bernoulliTemplate = struct( ...
    'mark', [], ...
    'existence', [], ...
    'particleWeights', [], ...
    'particlesKinematic', [], ...
    'particlesExtent', [], ...
    'particlesRate', []);

multiBernoulli = repmat(bernoulliTemplate,[0,1]);
meanUndetectedObjects = 0;

estimates = cell(numSteps,1);
elapsedTime = zeros(numSteps,1);
if nargout > 2
    pmbPosteriors = cell(numSteps,1);
end
for step = 1:numSteps

    stepTimer = tic;

    % Each step follows the standard PMB cycle:
    % 1) predict legacy Bernoullis and PPP intensity
    % 2) spawn candidate birth Bernoullis from "free" measurements
    % 3) run BP-based data association
    % 4) update, extract estimates, and prune weak Bernoullis
    estimates{step} = struct('mean',{},'extentMean',{});

    % Read the measurements for this scan.
    measurements = measurementsCell{step};
    numMeasurements = size(measurements,2);

    % Predict legacy Bernoullis and the undetected PPP mass.
    multiBernoulli = performBernoulliPrediction(multiBernoulli,parameters);
    numLegacy = numel(multiBernoulli);
    for target = 1:numLegacy
        probabilityEffectiveDetectPerParticle = 1 - exp(-multiBernoulli(target).particlesRate);
        averageProbabilityEffectiveDetect = ...
            multiBernoulli(target).particleWeights' * probabilityEffectiveDetectPerParticle;
        if averageProbabilityEffectiveDetect >= 1
            averageProbabilityEffectiveDetect = 1 - eps;
        end
        multiBernoulli(target).existence = multiBernoulli(target).existence * ...
            (1 - averageProbabilityEffectiveDetect) / ...
            (1 - multiBernoulli(target).existence * averageProbabilityEffectiveDetect);
    end
    meanUndetectedObjects = survivalProbability*meanUndetectedObjects+meanBirths;

    % Build candidate birth Bernoullis from measurements that remain weakly
    % explained by the legacy targets.
    [newIndexes,measurements,likelihood1] = getPromisingNewMeasurement(multiBernoulli,measurements,parameters);
    numNew = size(newIndexes,1);
    isNewIndex = false(1,numMeasurements);
    isNewIndex(newIndexes) = true;
    newMultiBernoulli = repmat(bernoulliTemplate,[numNew,1]);
    for target = 1:numNew
        proposalMean = measurements(:,newIndexes(target));
        newMultiBernoulli(target).particlesKinematic = zeros(4,numNewBornParticles);
        newMultiBernoulli(target).particlesKinematic(1:2,:) = ...
            proposalMean + proposalCovarianceChol * randn(2,numNewBornParticles);
        newMultiBernoulli(target).particlesKinematic(3:4,:) = ...
            priorVelocityCovarianceChol * randn(2,numNewBornParticles);
        newMultiBernoulli(target).particlesExtent = iwishrndFastVector( ...
            birthPriorExtent1,birthPriorExtent2,numNewBornParticles);
        newMultiBernoulli(target).particlesRate = gamrnd( ...
            birthPriorGammaAlpha,1/birthPriorGammaBeta,numNewBornParticles,1);
        proposalSamples = reshape( ...
            newMultiBernoulli(target).particlesKinematic(1:2,:),2,numNewBornParticles)';
        newMultiBernoulli(target).particleWeights = exp(normalizeLogWeights( ...
            uniformWeight - gaussianLogPdf(proposalSamples, proposalMean', proposalCovariance')));
        aliveProbability = meanUndetectedObjects * ...
            (newMultiBernoulli(target).particleWeights' * exp(-newMultiBernoulli(target).particlesRate));
        newMultiBernoulli(target).existence = aliveProbability/(aliveProbability+1);
        newMultiBernoulli(target).mark = [step;newIndexes(target)];
    end

    % Prepare per-target BP messages. New targets only connect to a prefix
    % of the reordered measurements, which keeps the update sparse.
    multiBernoulli = [multiBernoulli;newMultiBernoulli];
    currentExistenceExtrinsic = repmat([multiBernoulli.existence]',[1,numMeasurements]);
    numParticlesBernoulli = arrayfun(@(x) numel(x.particleWeights),multiBernoulli);
    weightsExtrinsic = cell(numLegacy,1);
    for target = 1:numLegacy
        weightsExtrinsic{target} = zeros(numParticlesBernoulli(target),numMeasurements);
    end
    weightsExtrinsicNew = cell(numNew,1);
    for target = 1:numNew
        weightsExtrinsicNew{target} = zeros(numParticlesBernoulli(numLegacy+target),numMeasurements);
    end
    likelihoodNew1 = weightsExtrinsicNew;

    for outer = 1:numOuterIterations
        if outer == 1
            targetIndex = numLegacy;
            for target = numMeasurements:-1:1
                if isNewIndex(target)
                    targetIndex = targetIndex + 1;
                    reshapedMean = reshape(multiBernoulli(targetIndex).particlesKinematic(1:2,:),2,numParticlesBernoulli(targetIndex));
                    reshapedCov = reshape(multiBernoulli(targetIndex).particlesExtent,2,2,numParticlesBernoulli(targetIndex)) + repmat(measurementsCovariance,[1,1,numParticlesBernoulli(targetIndex)]);
                    constantFactor = areaSize * (multiBernoulli(targetIndex).particlesRate / meanClutter);
                    logLikelihood = getLogWeightsFastMatrix(measurements,reshapedMean,reshapedCov);
                    likelihoodNew1{targetIndex-numLegacy} = bsxfun(@times, constantFactor, exp(logLikelihood));
                end
            end
        end

        % One BP sweep over measurement-to-target messages.
        outputDA = cell(numMeasurements,1);
        targetIndexes = cell(numMeasurements,1);
        for measurement = numMeasurements:-1:1
            inputDA = ones(2,numLegacy);
            for target = 1:numLegacy
                if(outer == 1)
                    inputDA(2,target) = currentExistenceExtrinsic(target,measurement) * multiBernoulli(target).particleWeights'*likelihood1{target}(:,measurement);
                else
                    inputDA(2,target) = currentExistenceExtrinsic(target,measurement) * (weightsExtrinsic{target}(:,measurement)'*likelihood1{target}(:,measurement));
                end
                inputDA(1,target) = 1;
            end
            targetIndex = numLegacy;
            targetIndexesCurrent = nan(numLegacy,1);

            % only new targets with index >= measurement index are connected to measurement
            for target = numMeasurements:-1:measurement
                if isNewIndex(target)
                    targetIndex = targetIndex + 1;
                    targetIndexesCurrent = [targetIndexesCurrent;target];
                    if(outer == 1)
                        inputDA(2,targetIndex) = currentExistenceExtrinsic(targetIndex,measurement) * (multiBernoulli(targetIndex).particleWeights'*likelihoodNew1{targetIndex-numLegacy}(:,measurement));
                    else
                        inputDA(2,targetIndex) = currentExistenceExtrinsic(targetIndex,measurement) * (weightsExtrinsicNew{targetIndex-numLegacy}(:,measurement)'*likelihoodNew1{targetIndex-numLegacy}(:,measurement));
                    end
                    inputDA(1,targetIndex) = 1;
                    if(target == measurement)
                        inputDA(1,targetIndex) = 1 - currentExistenceExtrinsic(targetIndex,measurement);
                    end
                end
            end
            targetIndexes{measurement} = targetIndexesCurrent;
            outputDA{measurement} = dataAssociationBP(inputDA);
        end

        % Update legacy Bernoullis.
        for target = 1:numLegacy
            weights = zeros(numParticlesBernoulli(target),numMeasurements+1);
            weights(:,numMeasurements+1) = log(multiBernoulli(target).particleWeights);
            for measurement = 1:numMeasurements
                currentWeights = 1 + likelihood1{target}(:,measurement) * outputDA{measurement}(1,target);
                currentWeights = log(currentWeights);
                weights(:,measurement) = currentWeights;
            end
            % calculate extrinsic information for legacy targets (at all except last iteration) and belief (at last iteration)
            if(outer ~= numOuterIterations)
                [weightsExtrinsic{target}, currentExistenceExtrinsic(target,:)] = ...
                    getWeightsUnknownNewAllMatrix(weights,multiBernoulli(target).existence,numMeasurements);
            else
                multiBernoulli(target) = updateParticlesNewBernoulli(multiBernoulli(target),weights);

                % Emit point/extent estimates before resampling destroys the
                % posterior particle weights.
                if(multiBernoulli(target).existence > detectionThreshold)
                    estimates{step}(end+1) = summarizeBernoulliEstimate(multiBernoulli(target));
                end

                % Keep the particle approximation well-conditioned for the
                % next prediction step.
                multiBernoulli(target) = resampleBernoulliTargets(multiBernoulli(target),numParticles);
            end
        end

        % Update candidate birth Bernoullis.
        targetIndex = numLegacy;
        for target = numMeasurements:-1:1
            if isNewIndex(target)
                targetIndex = targetIndex + 1;
                weights = zeros(numParticlesBernoulli(targetIndex),numMeasurements+1);
                weights(:,numMeasurements+1) = log(multiBernoulli(targetIndex).particleWeights);
                for measurement = 1:target
                    outputTmpDA = outputDA{measurement}(1,targetIndexes{measurement}==target);
                    if(~isinf(outputTmpDA))
                        currentWeights = likelihoodNew1{targetIndex-numLegacy}(:,measurement) * outputTmpDA;
                    else
                        currentWeights = likelihoodNew1{targetIndex-numLegacy}(:,measurement);
                    end
                    if(measurement ~= target)
                        currentWeights = currentWeights + 1;
                    end
                    currentWeights = log(currentWeights);
                    weights(:,measurement) = currentWeights;
                end

                % calculate extrinsic information for new targets (at all except last iteration) or belief (at last iteration)
                if(outer ~= numOuterIterations)
                    [weightsExtrinsicTmp, currentExistenceTmp] = ...
                        getWeightsUnknownNewAllMatrix(weights,multiBernoulli(targetIndex).existence,target);
                    weightsExtrinsicNew{targetIndex-numLegacy}(:,1:target) = weightsExtrinsicTmp;
                    currentExistenceExtrinsic(targetIndex,1:target) = currentExistenceTmp;
                else
                    multiBernoulli(targetIndex) = updateParticlesNewBernoulli(multiBernoulli(targetIndex),weights);
                    if numNewBornParticles > numParticles
                        multiBernoulli(targetIndex) = resampleBernoulliTargets(multiBernoulli(targetIndex),numParticles);
                        numParticlesBernoulli(targetIndex) = numParticles;
                    end
                end

                % New births are estimated from the same posterior summary as
                % legacy Bernoullis.
                if(multiBernoulli(targetIndex).existence > detectionThreshold)
                    estimates{step}(end+1) = summarizeBernoulliEstimate(multiBernoulli(targetIndex));
                end
            end
        end
    end

    % Drop Bernoullis that carry negligible existence probability.
    isNotRedundant = [multiBernoulli.existence] >= thresholdPruning;
    multiBernoulli = multiBernoulli(isNotRedundant);
    if nargout > 2
        pmbPosteriors{step} = summarizePMBPosterior(multiBernoulli);
    end

    % Advance the undetected PPP through a missed-detection update.
    meanUndetectedObjects = meanUndetectedObjects*probabilityPPPMissedDetect;

    elapsedTime(step) = toc(stepTimer);
end

runTime = sum(elapsedTime);

end
