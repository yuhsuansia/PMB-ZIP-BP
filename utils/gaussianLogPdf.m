function logDensity = gaussianLogPdf(samples, meanVector, covariance)
%GAUSSIANLOGPDF Gaussian log density with one common, full covariance.
%   Samples are N-by-D, meanVector is 1-by-D, and covariance is D-by-D.
%   This helper evaluates the birth proposal density in both BP filters.

dimension = size(samples, 2);
if ~isequal(size(meanVector), [1, dimension]) || ...
        ~isequal(size(covariance), [dimension, dimension])
    error('pmbbp:GaussianDimensions', 'Incompatible Gaussian input dimensions.');
end
factor = chol(covariance);
standardized = bsxfun(@minus, samples, meanVector) / factor;
logDensity = -0.5 * sum(standardized.^2, 2) - sum(log(diag(factor))) ...
    - dimension * log(2*pi) / 2;
end
