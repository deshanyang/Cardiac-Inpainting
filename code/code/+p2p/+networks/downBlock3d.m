function out = downBlock3d(id, nChannels, varargin)
% downBlock    Downsampling block

% Copyright 2020 The MathWorks, Inc.

    out = p2p.networks.block3d(id, nChannels, 'down', varargin{:});
    
end
