function testSmoke()
%TESTSMOKE Small deterministic integration and numerical checks.

root = fileparts(fileparts(mfilename('fullpath')));
previousPath = path;
previousRng = rng;
pathScope = onCleanup(@() path(previousPath)); %#ok<NASGU>
rngScope = onCleanup(@() rng(previousRng)); %#ok<NASGU>
addpath(root);
setupPath();

% Compare the proposal log density with the toolbox density for a correlated
% covariance, and confirm that log evaluation remains finite in the tails.
samples = [0, 0; 1, -2; 4, 3];
mu = [0.2, -0.1];
covariance = [2, 0.5; 0.5, 1];
actual = gaussianLogPdf(samples, mu, covariance);
reference = log(mvnpdf(samples, mu, covariance));
assert(max(abs(actual - reference)) < 1e-12);
assert(isfinite(gaussianLogPdf([1e3, -1e3], mu, covariance)));
assert(isequal(size(gaussianLogPdf(zeros(0,2), mu, covariance)), [0,1]));

% Clustered measurements create a birth; the next empty scan exercises
% prediction and missed detection for an existing Bernoulli.
cluster = [0, 0.4, -0.3; 0, 0.2, -0.2];
measurements = {cluster; zeros(2,0); cluster + 0.4; zeros(2,0)};
parameters = buildParameters(numel(measurements), 64, 3);
parameters.detectionThreshold = 0;
parameters = matchPredictionParameters(parameters);
for filter = {@eotEllipticalShapeRFS, @eotEllipticalShapeRFS_ZIP}
    rng(7, 'twister');
    [estimates, runTime, posteriors, elapsed] = filter{1}(measurements, parameters);
    assert(~isempty(posteriors{1}.r), 'The test must exercise a real birth.');
    verifyOutputs(estimates, runTime, posteriors, elapsed, numel(measurements));
    rng(7, 'twister');
    [rowEstimates, ~, rowPosteriors] = filter{1}(measurements.', parameters);
    assert(isequaln(estimates, rowEstimates) && isequaln(posteriors, rowPosteriors));
    [emptyEstimates, emptyTime, emptyPosteriors, emptyElapsed] = ...
        filter{1}({zeros(2,0); zeros(2,0)}, parameters);
    verifyOutputs(emptyEstimates, emptyTime, emptyPosteriors, emptyElapsed, 2);
    assert(all(cellfun(@isempty, emptyEstimates)));
end

% Both detection-probability endpoints should yield finite ZIP posteriors.
for detectionProbability = [0, 1]
    parameters.detectionProbability = detectionProbability;
    rng(7, 'twister');
    [estimates, runTime, posteriors, elapsed] = ...
        eotEllipticalShapeRFS_ZIP(measurements, parameters);
    verifyOutputs(estimates, runTime, posteriors, elapsed, numel(measurements));
end

options = struct('numSteps', 6, 'numParticles', 64, 'makePlots', false);
for demo = {@demoPoisson, @demoZIP}
    startingPath = path;
    startingRng = rng;
    result = demo{1}(options);
    assert(strcmp(path, startingPath), 'Demo changed the caller path.');
    assert(isequal(rng, startingRng), 'Demo changed the caller random stream.');
    verifyOutputs(result.estimates, result.runTime, result.pmbPosteriors, ...
        result.elapsedTime, options.numSteps);
end
fprintf('All PMB-BP smoke checks passed.\n');
end

function verifyOutputs(estimates, runTime, posteriors, elapsed, numSteps)
assert(isequal(size(estimates), [numSteps,1]));
assert(isequal(size(posteriors), [numSteps,1]));
assert(isequal(size(elapsed), [numSteps,1]));
assert(all(isfinite(elapsed)) && all(elapsed >= 0));
assert(isfinite(runTime) && runTime == sum(elapsed));
for step = 1:numSteps
    posterior = posteriors{step};
    assert(all(isfinite(posterior.r)) && all(posterior.r >= 0 & posterior.r <= 1));
    assert(all(isfinite(posterior.x(:))) && all(isfinite(posterior.X(:))));
    assert(size(posterior.x,2) == numel(posterior.r));
    for object = 1:numel(posterior.r)
        extent = posterior.X(:,:,object);
        assert(norm(extent - extent', 'fro') < 1e-9);
        assert(all(eig(extent) > 0));
    end
    for object = 1:numel(estimates{step})
        estimate = estimates{step}(object);
        assert(all(isfinite(estimate.mean)) && all(isfinite(estimate.extentMean(:))));
    end
end
end
