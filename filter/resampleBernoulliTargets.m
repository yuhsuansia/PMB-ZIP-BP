function Bernoulli = resampleBernoulliTargets(Bernoulli,numParticles)
%RESAMPLEBERNOULLITARGETS Resample all state components with common indices.

indexes = resampleSystematic(Bernoulli.particleWeights,numParticles);
Bernoulli.particleWeights = 1/numParticles*ones(numParticles,1);
Bernoulli.particlesKinematic = Bernoulli.particlesKinematic(:,indexes);
Bernoulli.particlesExtent = Bernoulli.particlesExtent(:,:,indexes);
Bernoulli.particlesRate = Bernoulli.particlesRate(indexes);

end
