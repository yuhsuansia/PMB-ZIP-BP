function parameters = buildParameters(numSteps, numParticles, numOuterIterations)
%BUILDPARAMETERS Build default simulation and filter parameters.

parameters.numSteps = numSteps;

% Main parameters of the statistical model.
parameters.scanTime = 0.2;
parameters.accelerationDeviation = 0.8;
parameters.survivalProbability = 0.99;
parameters.meanBirths = 0.01;
parameters.surveillanceRegion = [[-150; 150] [-150; 150]];
parameters.measurementVariance = 0;
parameters.meanMeasurements = 5;
parameters.detectionProbability = 0.9;
parameters.meanClutter = 10;
parameters.truthSeed = 0;

% Prior distribution parameters.
parameters.priorVelocityCovariance = diag([10^2;10^2]);
parameters.priorExtent2 = 100;
parameters.priorExtent1 = diag([5 5])*(parameters.priorExtent2-3);
parameters.degreeFreedomPrediction1 = 10000;
parameters.priorGammaBeta = 100;
parameters.priorGammaAlpha = parameters.meanMeasurements*parameters.priorGammaBeta;
parameters.degreeFreedomPrediction2 = 10000;
parameters.predictionGammaEta = 1.01;
parameters.predictionExtentTau = parameters.scanTime * 100;

% Birth density parameters.
parameters.birthPriorExtent2 = 100;
parameters.birthPriorExtent1 = diag([5 5])*(parameters.birthPriorExtent2-3);
parameters.birthPriorGammaBeta = 100;
parameters.birthPriorGammaAlpha = parameters.meanMeasurements*parameters.birthPriorGammaBeta;

% Sampling parameters.
parameters.numParticles = numParticles;
parameters.numNewBornParticles = 4*parameters.numParticles;
parameters.regularizationDeviation = 0;

% Detection and pruning parameters.
parameters.detectionThreshold = 0.5;
parameters.thresholdPruning = 10^(-3);

% Censoring and measurement reordering parameters.
parameters.freeThreshold = 0.95;
parameters.clusterThreshold = 0.9;
parameters.minClusterElements = 2;
parameters.enableProbabilityThresholding = true;
parameters.enableMeasurementClustering = true;
parameters.enableMeasurementReordering = true;

% Message passing parameters.
parameters.numOuterIterations = numOuterIterations;

% GOSPA parameters.
parameters.GOSPA_p = 1;
parameters.GOSPA_c = 20;

end
