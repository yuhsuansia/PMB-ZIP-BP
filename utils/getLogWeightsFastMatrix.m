function logWeights = getLogWeightsFastMatrix(measurements, currentParticlesKinematic, currentParticlesExtent)
%GETLOGWEIGHTSFASTMATRIX Vectorized Gaussian log-likelihoods for all measurements.
numParticles = size(currentParticlesExtent,3);
numMeasurements = size(measurements,2);

if numParticles == 0 || numMeasurements == 0
    logWeights = zeros(numParticles,numMeasurements);
    return;
end

extent11 = reshape(currentParticlesExtent(1,1,:),numParticles,1);
extent12 = 0.5 * (reshape(currentParticlesExtent(1,2,:),numParticles,1) + ...
                   reshape(currentParticlesExtent(2,1,:),numParticles,1));
extent22 = reshape(currentParticlesExtent(2,2,:),numParticles,1);

allDeterminantes = extent11.*extent22 - extent12.^2;
allFactors = -log(2*pi) - 0.5*log(allDeterminantes);

inv11 = extent22 ./ allDeterminantes;
inv12 = -extent12 ./ allDeterminantes;
inv22 = extent11 ./ allDeterminantes;

deltaX = bsxfun(@minus, measurements(1,:), currentParticlesKinematic(1,:)');
deltaY = bsxfun(@minus, measurements(2,:), currentParticlesKinematic(2,:)');

quadraticForm = bsxfun(@times, inv11, deltaX.^2) + ...
    2*bsxfun(@times, inv12, deltaX.*deltaY) + ...
    bsxfun(@times, inv22, deltaY.^2);

logWeights = bsxfun(@minus, allFactors, 0.5*quadraticForm);
end
