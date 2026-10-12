function result = demoPoisson(options)
%DEMOPOISSON Run one simulated Poisson-measurement PMB-BP example.
%   result = demoPoisson(struct('numSteps', 30, 'makePlots', false));

if nargin < 1
    options = struct();
end
scope = setupPath(); %#ok<NASGU>
result = runDemo('poisson', options);
end
