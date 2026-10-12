function [log_w,log_sum_w] = normalizeLogWeightsMatrix(log_w)
%NORMALIZELOGWEIGHTSMATRIX Normalize each column of a log-weight matrix.

if isempty(log_w)
    log_sum_w = -inf(1,size(log_w,2));
    return;
end

log_w(isnan(log_w)) = -inf;

numCols = size(log_w,2);
log_sum_w = -inf(1,numCols);
log_w_normalized = -inf(size(log_w));

positiveInfMask = isinf(log_w) & (log_w > 0);
numPositiveInf = sum(positiveInfMask,1);
colsPositiveInf = find(numPositiveInf > 0);
for col = colsPositiveInf
    % If a column contains +Inf entries, all finite entries are dominated.
    log_sum_w(col) = inf;
    log_w_normalized(positiveInfMask(:,col),col) = -log(numPositiveInf(col));
end

allNegativeInf = all(log_w == -inf,1);
colsFinite = ~(numPositiveInf > 0 | allNegativeInf);
if any(colsFinite)
    % Standard log-sum-exp normalization for the remaining finite columns.
    maxLogWeights = max(log_w(:,colsFinite),[],1);
    shiftedWeights = bsxfun(@minus, log_w(:,colsFinite), maxLogWeights);
    log_sum_w(colsFinite) = maxLogWeights + log(sum(exp(shiftedWeights),1));
    log_w_normalized(:,colsFinite) = bsxfun(@minus, log_w(:,colsFinite), log_sum_w(colsFinite));
end

log_w = log_w_normalized;
end
