function multiBernoulli = performBernoulliPrediction(multiBernoulli, parameters)
%PERFORMBERNOULLIPREDICTION Predict PMB Bernoulli components one step ahead.
numTargets = numel(multiBernoulli);
drivingNoiseVariance = parameters.accelerationDeviation^2;
survivalProbability = parameters.survivalProbability;
degreeFreedomPrediction1 = parameters.degreeFreedomPrediction1;
degreeFreedomPrediction2 = parameters.degreeFreedomPrediction2;
scanTime = parameters.scanTime;

[A, Q] = getTransitionMatricesNew(scanTime);
kinematicNoiseChol = chol(drivingNoiseVariance*Q,'lower');

for target = 1:numTargets
    numParticles = numel(multiBernoulli(target).particleWeights);
    multiBernoulli(target).existence = survivalProbability*multiBernoulli(target).existence;
    multiBernoulli(target).particlesKinematic = A*multiBernoulli(target).particlesKinematic + ...
        kinematicNoiseChol*randn(4,numParticles);
    multiBernoulli(target).particlesExtent = wishrndFastVector(multiBernoulli(target).particlesExtent/degreeFreedomPrediction1,degreeFreedomPrediction1,numParticles);
    multiBernoulli(target).particlesRate = gamrnd( ...
        multiBernoulli(target).particlesRate*degreeFreedomPrediction2, ...
        1/degreeFreedomPrediction2, numParticles, 1);
end

end
