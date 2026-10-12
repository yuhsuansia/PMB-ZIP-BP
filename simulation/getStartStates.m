function [startStates, startMatrixes, startMeasurementRates] = getStartStates(numObjects, radius, speed, parameters)
%GETSTARTSTATES Sample initial kinematic and extent states.

nu  = parameters.priorExtent2;
Psi = parameters.priorExtent1;

startMatrixes = zeros(2,2,numObjects);
for k = 1:numObjects
    startMatrixes(:,:,k) = iwishrnd(Psi, nu);
end

startMeasurementRates = gamrnd(parameters.priorGammaAlpha, 1/parameters.priorGammaBeta, numObjects, 1);

if numObjects < 2
    startStates = zeros(4,1);
    startStates(3) = speed;
    return
end

angles = (0:numObjects-1) * (2*pi/numObjects);
sa = sin(angles);
ca = cos(angles);

startStates = [ ...
    sa * radius; ...
    ca * radius; ...
    -sa * speed; ...
    -ca * speed ];

startStates(:,1) = [0; radius; 0; -speed];
end
