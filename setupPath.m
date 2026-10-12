function cleanup = setupPath()
%SETUPPATH Add only this package's implementation and example folders.
%   setupPath() keeps the added paths for interactive use.
%   cleanup = setupPath() restores the previous path when cleanup is cleared.

previousPath = path;
root = fileparts(mfilename('fullpath'));
folders = {'filter', 'simulation', 'utils', 'examples'};
addpath(root);
for index = 1:numel(folders)
    addpath(fullfile(root, folders{index}));
end
if nargout > 0
    cleanup = onCleanup(@() path(previousPath));
end
end
