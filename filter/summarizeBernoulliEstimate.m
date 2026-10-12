function estimate = summarizeBernoulliEstimate(Bernoulli)
%SUMMARIZEBERNOULLIESTIMATE Weighted kinematic and extent posterior means.
particleWeights = Bernoulli.particleWeights(:);
numParticles = numel(particleWeights);
particleWeightReshape = reshape(particleWeights,1,1,numParticles);

estimate = struct( ...
    'mean', sum(Bernoulli.particlesKinematic.*particleWeights',2), ...
    'extentMean', sum(Bernoulli.particlesExtent.*particleWeightReshape,3));
end
