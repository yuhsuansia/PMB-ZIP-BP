function fig = plotDemo(result)
%PLOTDEMO Show one run's trajectories, measurements, estimates and counts.

fig = figure('Name', ['PMB-BP: ' upper(result.model)], 'Color', 'w');
layout = tiledlayout(fig, 1, 2);
ax = nexttile(layout);
hold(ax, 'on');
measurements = [result.data.measCell{:}];
hMeasurements = plot(ax, measurements(1,:), measurements(2,:), '.', ...
    'Color', [0.8, 0.8, 0.8], 'MarkerSize', 3);
tracks = result.data.truthTracks;
for object = 1:result.data.numObjects
    plot(ax, tracks(1,:,object), tracks(2,:,object), '-', 'Color', [0.2, 0.4, 0.7]);
end
positions = zeros(2, 0);
for step = 1:numel(result.estimates)
    current = result.estimates{step};
    if ~isempty(current)
        means = [current.mean];
        positions = [positions, means(1:2,:)]; %#ok<AGROW>
    end
end
hEstimates = plot(ax, positions(1,:), positions(2,:), '.', ...
    'Color', [0.85, 0.25, 0.15], 'MarkerSize', 5);
hTruth = plot(ax, nan, nan, '-', 'Color', [0.2, 0.4, 0.7]);
legend(ax, [hMeasurements, hTruth, hEstimates], {'Measurements', 'Truth', 'Estimates'});
axis(ax, 'equal');
grid(ax, 'on');
xlabel(ax, 'x (m)'); ylabel(ax, 'y (m)');
title(ax, 'Positions over the full run');

ax = nexttile(layout);
truthCounts = sum(~isnan(tracks(1,:,:)), 3);
estimateCounts = cellfun(@numel, result.estimates);
plot(ax, 1:numel(truthCounts), truthCounts(:), '-', ...
    1:numel(estimateCounts), estimateCounts, '--', 'LineWidth', 1.2);
grid(ax, 'on');
xlabel(ax, 'Scan'); ylabel(ax, 'Object count');
legend(ax, {'Truth', 'Estimates'});
title(ax, 'Number of objects');
end
