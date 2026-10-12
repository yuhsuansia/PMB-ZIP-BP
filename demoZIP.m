function result = demoZIP(options)
%DEMOZIP Run one simulated zero-inflated Poisson PMB-BP example.
%   result = demoZIP(struct('detectionProbability', 0.8, 'makePlots', false));

if nargin < 1
    options = struct();
end
scope = setupPath(); %#ok<NASGU>
result = runDemo('zip', options);
end
