classdef TrainingPlot < handle
% TrainingPlot    Displays training progress

% Copyright 2020 The MathWorks, Inc.
    
    properties (Access = private)
        TiledChart
        InputsAx
        OutputsAx
        LossAx1
        LossAx2
        InputsIm
        OutputsIm
        ExampleInputs
        NumExampleInputsGroups
        curExampleGroupNum
        Lines = matlab.graphics.animation.AnimatedLine.empty
        StartTime
    end
    
    methods
        function obj = TrainingPlot(exampleInputs)
            
            obj.StartTime = datetime("now");
            
            trainingName = sprintf("pix2pix training started at %s", ...
                                obj.StartTime);
            fig = figure("Units", "Normalized", ...
                            "Position", [0.1, 0.1, 0.7, 0.6], ...
                            "Name", trainingName, ...
                            "NumberTitle", "off", ...
                            "Tag", "p2p.vis.TrainingPlot");
            obj.TiledChart = tiledlayout(fig, 3, 4, ...
                                        "TileSpacing", "compact", ...
                                        "Padding", "compact");
            obj.InputsAx = nexttile(obj.TiledChart, 1, [2, 2]);
            obj.OutputsAx = nexttile(obj.TiledChart, 3, [2, 2]);
            obj.LossAx1 = nexttile(obj.TiledChart, 9, [1, 2]);
            obj.LossAx2 = nexttile(obj.TiledChart, 11, [1, 2]);
            
            obj.ExampleInputs = exampleInputs;
            obj.NumExampleInputsGroups = floor(size(exampleInputs,5)/9);
            obj.curExampleGroupNum = 1;
            obj.initImages();
            obj.initLines();
            drawnow();
        end
        
        function update(obj, epoch, iteration,  ...
                    gLoss, lossL1, ganLoss, dLoss, generator)
            obj.updateImages(generator)
            obj.updateLines(epoch, iteration, gLoss, lossL1, ganLoss, dLoss);
            drawnow();
        end
        
        function initImages(obj)
            exampleIdxes = (obj.curExampleGroupNum-1)*9+(1:9);
            artifact_mask = obj.ExampleInputs(:,:,:,2,exampleIdxes)>0;
            displayIm = obj.prepForPlot(obj.ExampleInputs(:,:,:,1,exampleIdxes), artifact_mask);
            montageIm = imtile(displayIm);
            obj.InputsIm = imshow(montageIm, "Parent", obj.InputsAx);
            
            zeroIm = 1*montageIm;
            obj.OutputsIm = imshow(zeroIm, "Parent", obj.OutputsAx);
            
            set(obj.InputsAx, 'clim', [0.4 0.6]);
            set(obj.OutputsAx, 'clim', [0.4 0.6]);
            % set(obj.InputsAx, 'clim', [0.1 0.6]);
            % set(obj.OutputsAx, 'clim', [0.1 0.6]);
        end
        
        function updateImages(obj, generator)
            obj.curExampleGroupNum = randi(obj.NumExampleInputsGroups);
            exampleIdxes = (obj.curExampleGroupNum-1)*9+(1:9);
            curExampleInputs=obj.ExampleInputs(:,:,:,:,exampleIdxes);
            output = tanh(generator.forward(curExampleInputs));

            % To focus only in the band. This is how the network is trained.
            % band_mask = obj.ExampleInputs==min(obj.ExampleInputs(:));
            % 
            % band_mask = curExampleInputs==min(curExampleInputs(:));
            % output(band_mask==0) = curExampleInputs(band_mask==0);

            curExampleInputs2 = curExampleInputs(:,:,:,1,:);
            curExampleMaskInputs = curExampleInputs(:,:,:,2,:);
            artifact_mask = curExampleMaskInputs>0;
            output(artifact_mask==0) = curExampleInputs2(artifact_mask==0);

            [displayIm, slice_idx] = obj.prepForPlot(output, artifact_mask);
            obj.OutputsIm.CData = imtile(displayIm);

            displayIm2 = obj.prepForPlot(curExampleInputs2, artifact_mask, slice_idx);
            obj.InputsIm.CData = imtile(displayIm2);            
        end
        
        function initLines(obj)
            % First plot just for generator
            obj.Lines(1) = animatedline(obj.LossAx1, ...
                                        "LineWidth", 1, ...
                                        "DisplayName", "Generator total");
            xlabel(obj.LossAx1, "Iteration");
            ylabel(obj.LossAx1, "Loss");
            grid(obj.LossAx1, "on");
            legend(obj.LossAx1);
            
            % Remaining plots for other losses
            nLines = 3;
            cMap = lines(nLines);
            labels = ["L1 loss", "GAN loss", "Discriminator loss"];
            for iLine = 1:nLines
                obj.Lines(iLine + 1) = animatedline(obj.LossAx2, ...
                                                "Color", cMap(iLine, :), ...
                                                "LineWidth", 1, ...
                                                "DisplayName", labels(iLine));
            end
            xlabel(obj.LossAx2, "Iteration");
            ylabel(obj.LossAx2, "Loss");
            grid(obj.LossAx2, "on");
            legend(obj.LossAx2);
        end
        
        function updateLines(obj, epoch, iteration, gLoss, lossL1, ganLoss, dLoss)
            titleString = sprintf("Current epoch: %d, elapsed time: %s", ...
                                    epoch, datetime("now") - obj.StartTime);
            title(obj.LossAx1, titleString);
            addpoints(obj.Lines(1), iteration, double(gLoss));
            % if lossL1<1e-1
            %     addpoints(obj.Lines(2), iteration, double(lossL1*20));
            if lossL1<1e-2
                addpoints(obj.Lines(2), iteration, double(lossL1*200));
            else
                addpoints(obj.Lines(2), iteration, double(lossL1));
            end
            addpoints(obj.Lines(3), iteration, double(ganLoss));
            addpoints(obj.Lines(4), iteration, double(dLoss));
        end
        
    end
    
    methods (Static)
        function [imOut, slice_idx] = prepForPlot(im, artifact_mask, slice_idx)
            imOut0 = (gather(extractdata(im)) + 1)/2;
            
            % only take the first channel for n != 3
            if ~exist('slice_idx', 'var')
                mask=artifact_mask>0;
                val = sum(mask, [1 2 4 5]);
                [~,best_slice] = max(val);
                slice_idx = gather(extractdata(best_slice));
            end

            imOut = imOut0(:,:,slice_idx,:);
            for k=1:size(im,5)
                mask = artifact_mask(:,:,:,:,k)>0;
                val = sum(mask, [1 2 4]);
                [~,best_slice] = max(val);
                imOut(:,:,:,k) = imOut0(:,:,best_slice,:,k);
            end
        end
    end
end
