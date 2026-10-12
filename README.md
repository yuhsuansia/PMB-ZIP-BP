# Multiple extended object tracking using the PMB filter with the zero-inflated Poisson point process model

MATLAB implementation of an extended object Poisson multi-Bernoulli (PMB) filter with the zero-inflated Poisson point process measurement model using particle belief propagation. Two measurement models
are included:

- **Poisson:** each object generates a Poisson measurement set every scan.
- **Zero-inflated Poisson (ZIP):** an object is detected with probability `pD`;
  conditional on detection its measurement count is Poisson. A detected object
  can still generate zero measurements.

The state includes 2D position/velocity, a 2-by-2 extent covariance, and a
measurement rate. The filters use particle Bernoulli components, uniform
spatial births.

## Paper

1. An Extended Object PMB Filter with Zero-Inflated Poisson Model Using Belief Propagation, 2026 IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI).
2. Extended version available at https://arxiv.org/abs/2605.25316.

## Quick start

Open this folder in MATLAB and run:

```matlab
result = demoZIP();      % One ZIP run with pD = 0.8 and a plot
result = demoPoisson();  % One Poisson run with a plot
```

For a shorter run or different settings:

```matlab
result = demoZIP(struct('numSteps', 30, 'numParticles', 500, ...
    'numOuterIterations', 3, 'detectionProbability', 0.8, ...
    'measurementRate', 8, 'meanClutter', 10, 'seed', 1, 'makePlots', false));
```

Each demo creates its own synthetic measurements and returns the configuration,
truth, measurements, estimates, posterior moments, and timings. It restores
the caller's MATLAB path and random stream. Truth uses seed 0; the `seed`
option controls measurement generation and filter randomness. No files are
saved automatically. The Poisson demo uses detection probability 1.

## Using your own measurements

```matlab
setupPath();
T = numel(measurements);
parameters = buildParameters(T, 1000, 3);
parameters.detectionProbability = 0.8;
parameters = matchPredictionParameters(parameters);
[estimates, runTime, pmbPosteriors, elapsedTime] = ...
    eotEllipticalShapeRFS_ZIP(measurements, parameters);
```

`measurements` is a cell vector with one real 2-by-M measurement matrix per
scan. Use `zeros(2,0)` for an empty scan. For the Poisson model, call
`eotEllipticalShapeRFS` instead. All scan outputs are column vectors.

- `estimates{k}` is a struct array with `mean` (4-by-1 position/velocity) and
  `extentMean` (2-by-2 covariance) for reported objects.
- `pmbPosteriors{k}` contains `r` (existence probabilities), `x` (2-by-N
  position means), and `X` (2-by-2-by-N extent means), after pruning.
  These are moment summaries, not the full particle posterior.
- `elapsedTime(k)` measures filtering for scan k; `runTime` is their sum.

Adjust model and birth priors in `simulation/buildParameters.m`. If changing
the measurement rate, update both `priorGammaAlpha` and `birthPriorGammaAlpha`
to rate times their corresponding beta. The default measurement variance is
zero; the supplied synthetic generator uses extent covariance alone.
The clutter mean must be positive because likelihood ratios divide by it.

## Layout

```text
demoPoisson.m, demoZIP.m  Single-run entry points
setupPath.m              Explicit package path setup
examples/                Shared demo setup and plotting
filter/                  BP filtering, prediction, resampling and summaries
simulation/              Synthetic scenarios and parameter setup
utils/                   Likelihood, clustering and numerical helpers
tests/                   Small deterministic smoke and numerical checks
```

## Verification

```matlab
addpath('tests');
testSmoke();
```

The checks exercise both models, empty scans, posterior validity, demo path/RNG
restoration, and Gaussian proposal densities. They run short examples without
writing result files.
