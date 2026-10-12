function pmb = summarizePMBPosterior(multiBernoulli)
%SUMMARIZEPMBPOSTERIOR Existence probabilities and position/extent moments.
numBernoulli = numel(multiBernoulli);
r = zeros(1,numBernoulli);
x = zeros(2,numBernoulli);
X = zeros(2,2,numBernoulli);
count = 0;

for i = 1:numBernoulli
    rs = multiBernoulli(i).existence;
    if isempty(rs) || ~isfinite(rs) || rs <= 0
        continue
    end

    weights = multiBernoulli(i).particleWeights(:);
    if isempty(weights)
        continue
    end

    particlesKinematic = multiBernoulli(i).particlesKinematic;
    particlesExtent = multiBernoulli(i).particlesExtent;
    numParticles = numel(weights);
    if size(particlesKinematic,2) < numParticles || size(particlesExtent,3) < numParticles
        continue
    end

    valid = isfinite(weights) & (weights >= 0);
    if ~any(valid)
        continue
    end
    weights(~valid) = 0;

    particlesKinematic = particlesKinematic(1:2,1:numParticles);
    for j = 1:numParticles
        extentCurrent = particlesExtent(:,:,j);
        if ~all(isfinite(particlesKinematic(:,j))) || ~all(isfinite(extentCurrent(:)))
            weights(j) = 0;
        end
    end

    weightSum = sum(weights);
    if weightSum <= 0
        continue
    end
    weights = weights / weightSum;

    meanKinematic = particlesKinematic * weights;
    meanExtent = zeros(2,2);
    for j = 1:numParticles
        meanExtent = meanExtent + weights(j) * particlesExtent(:,:,j);
    end

    if ~all(isfinite(meanKinematic)) || ~all(isfinite(meanExtent(:)))
        continue
    end

    count = count + 1;
    r(count) = rs;
    x(:,count) = meanKinematic;
    X(:,:,count) = meanExtent;
end

pmb = struct('r', r(1:count), 'x', x(:,1:count), 'X', X(:,:,1:count));
end
