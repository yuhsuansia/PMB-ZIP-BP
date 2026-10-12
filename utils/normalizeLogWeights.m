function [log_w,log_sum_w] = normalizeLogWeights(log_w)
%NORMALIZELOGWEIGHTS Normalize a log-weight vector using log-sum-exp.
%   The second output is the log of the unnormalized weight sum.

if isempty(log_w)
    log_sum_w = -inf;
    return;
end

log_w(isnan(log_w)) = -inf;

positiveInfMask = isinf(log_w) & (log_w > 0);
if any(positiveInfMask(:))
    numPositiveInf = sum(positiveInfMask(:));
    log_sum_w = inf;
    log_w(~positiveInfMask) = -inf;
    log_w(positiveInfMask) = -log(numPositiveInf);
    return;
end

if all(log_w(:) == -inf)
    log_sum_w = -inf;
elseif numel(log_w)<=1
    % Log of sum of weights times prior probabilities
    log_sum_w = log_w;
    % Normalize
    log_w = log_w-log_sum_w;
else
    % Sum the remaining terms relative to the largest log weight.
    I = true(numel(log_w),1);
    [max_log_w_aux,idx] = max(log_w);
    I(idx) = false;
    log_sum_w = max_log_w_aux+log(1+sum(exp(log_w(I)-max_log_w_aux)));
    % Normalize
    log_w = log_w-log_sum_w;
end
