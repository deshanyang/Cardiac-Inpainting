function model = discriminator3d(inputSize, inputChannels, depth)
% discriminator    pix2pix discriminator network

% Copyright 2020 The MathWorks, Inc.
        
    layers = image3dInputLayer([inputSize, inputChannels], 'Name', 'inputImage', 'Normalization', 'none');
    
    downChannels = [64, 128, 256, 512];
    
    if nargin < 3
        depth = numel(downChannels);
    else
        assert(depth <= 4, ...
            'p2p:networks:discriminator', ...
            'Current max depth is 4');
        % Modify down and up channels accordingly
        downChannels = downChannels(1:depth);
    end
    
    for iLevel = 1:depth
        
        if iLevel == 1
            doNorm = false;
        else
            doNorm = true;
        end
        
        layers = [layers
            p2p.networks.downBlock3d(sprintf('D_%d', iLevel), downChannels(iLevel), ...
                            'DoNorm', doNorm)];
    end
    
    layers = [layers
        convolution3dLayer(1, 1, ...
                            'Name', 'outputLayer', ...
                            'Padding', 'same', ...
                            'Stride', 1, ...
                            'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'))];
    lg = layerGraph(layers);
    model = dlnetwork(lg);
end