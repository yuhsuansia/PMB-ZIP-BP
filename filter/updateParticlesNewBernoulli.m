function Bernoulli = updateParticlesNewBernoulli(Bernoulli,logWeights)
%UPDATEPARTICLESNEWBERNOULLI Normalize a Bernoulli's final particle belief.

logWeights = sum(logWeights,2);
[log_w,log_sum_w] = normalizeLogWeights(logWeights);

if isinf(log_sum_w) && log_sum_w > 0
    Bernoulli.existence = 1;
elseif isinf(log_sum_w) && log_sum_w < 0
    Bernoulli.existence = 0;
else
    aliveUpdate = exp(log_sum_w);
    alive = Bernoulli.existence*aliveUpdate;
    dead = (1-Bernoulli.existence);
    denominator = dead + alive;
    if denominator <= 0
        Bernoulli.existence = double(alive > 0);
    else
        Bernoulli.existence = alive/denominator;
    end
end
Bernoulli.existence = min(max(Bernoulli.existence,0),1);

if(Bernoulli.existence > 0 && any(isfinite(log_w)))
    Bernoulli.particleWeights = exp(log_w);
else
    Bernoulli.particleWeights = nan(size(logWeights));
    Bernoulli.particlesKinematic = nan(size(Bernoulli.particlesKinematic));
    Bernoulli.particlesExtent = nan(size(Bernoulli.particlesExtent));
end

end
