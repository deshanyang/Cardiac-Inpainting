classdef PairedImageDatastore_Cardiac < matlab.io.Datastore & ...
                                matlab.io.datastore.Shuffleable & ...
                                matlab.io.datastore.MiniBatchable
% PairedImageDatastore A datastore to provide pairs of images.
%
%   This datastore allows mini-batching and shuffling of matching pairs of
%   images in two folders while, preserving the pairing of images.

% Copyright 2020 The MathWorks, Inc.

    properties (Dependent)
        MiniBatchSize
    end
    
    properties (SetAccess = protected)
        DirA
        DirB
        DirC
        imagesA
        imagesA_cache
        imagesB
        imagesB_cache
        points
        points_cache
        NumObservations
        MiniBatchSize_
        Augmenter
        PreSize
        CropSize
        ARange
        BRange
        data_indices
        data_current_index
    end
     
    methods (Static)
        function [inputs, remaining] = parseInputs(varargin)
            parser = inputParser();
            % Remaining inputs should be for the imageAugmenter
            parser.KeepUnmatched = true;
            parser.addParameter('PreSize', [256, 256]);
            parser.addParameter('CropSize', [256, 256]);
            parser.addParameter('ARange', 255);
            parser.addParameter('BRange', 255);
            parser.parse(varargin{:});
            inputs = parser.Results;
            remaining = parser.Unmatched;
        end
    end
    
    methods
        function obj = PairedImageDatastore_Cardiac(dirA, dirB, dirC, miniBatchSize, varargin)
            % Create a PairedImageDatastore
            %
            % Args:
            %   dirA            - directory or cell array of filenames
            %   dirB            - directory or cell array of filenames
            %   dirC            - directory or cell array of filenames
            %   miniBatchSize   - Number of image pairs to provide in each
            %                       minibatch
            % TODO list optional name-value pairs PreSize, CropSize,
            % Mirror
            %
            % Note:
            %   This datastore relies on the naming of image files in the
            %   two directory to appear in the same ordering for correct
            %   pairing. The simplest way to ensure this is if pairs of
            %   imagesA both have the same name.
            
            includeSubFolders = true;
            
            obj.DirA = dirA;
            obj.DirB = dirB;
            obj.DirC = dirC;
            obj.imagesA = imageDatastore(obj.DirA, "IncludeSubfolders", includeSubFolders, 'FileExtensions', '.mat', 'ReadFcn', @mat_reader);
            obj.imagesB = imageDatastore(obj.DirB, "IncludeSubfolders", includeSubFolders, 'FileExtensions', '.mat', 'ReadFcn', @mat_reader);
            obj.points = imageDatastore(obj.DirC, "IncludeSubfolders", includeSubFolders, 'FileExtensions', '.mat', 'ReadFcn', @mat_reader);
            obj.MiniBatchSize = miniBatchSize;
            
            assert(numel(obj.imagesA.Files) == numel(obj.points.Files), ...
                    'p2p:datastore:notMatched', ...
                    'Number of files in A and B folders do not match');
            obj.NumObservations = numel(obj.imagesA.Files);

            fprintf('Reading all %d datasets into RAM ... ', obj.NumObservations);
            obj.points_cache = obj.points.readall();
            obj.imagesA_cache = obj.imagesA.readall();
            obj.imagesB_cache = obj.imagesB.readall();
            fprintf('Done ...\n');
            obj.data_indices = 1:obj.NumObservations;
            obj.data_current_index = 1;
            
            % Handle optional arguments
            [inputs, remaining] = obj.parseInputs(varargin{:});
            
            obj.ARange = inputs.ARange;
            obj.BRange = inputs.BRange;
            obj.Augmenter = imageDataAugmenter(remaining);
            obj.PreSize = inputs.PreSize;
            obj.CropSize = inputs.CropSize;
        end
        
        function tf = hasdata(obj)
            tf = obj.data_current_index <= (obj.NumObservations - obj.MiniBatchSize)*100;
            if ~tf
                obj.data_current_index=1;
            end
            % tf = true;
            % tf = obj.imagesA.hasdata() && obj.points.hasdata();
        end
        
        function data = read(obj)
            data_indices_2_read = (1:obj.MiniBatchSize)+obj.data_current_index-1;
            data_indices_2_read = mod(data_indices_2_read-1, obj.NumObservations)+1;
            obj.data_current_index = obj.data_current_index + obj.MiniBatchSize;

            imgsA = obj.imagesA_cache(data_indices_2_read);
            imgsB = obj.imagesB_cache(data_indices_2_read);
            cpts = obj.points_cache(data_indices_2_read);
            
            % for batch size 1 imagedatastore doesn't wrap in a cell
            if ~iscell(imgsA)
                imgsA = {imgsA};
                imgsB = {imgsB};
                cpts = {cpts};
            end

            [patchesA, patchesB] = sample_and_mask(imgsA, imgsB, cpts, obj.CropSize);

            % Disable argumenters. 3D is not supported for now.
            % [transformedA, transformedB] = ...
            %     p2p.data.transformImagePair(patchesA, patchesB, ...
            %     obj.PreSize, obj.CropSize, ...
            %     obj.Augmenter);
            % [A, B] = obj.normaliseImages(transformedA, transformedB);

            [A, B] = obj.normaliseImages(patchesA, patchesB);
            data = table(A, B);
        end

        function reset(obj)
            obj.data_current_index = 1;
            % obj.imagesA.reset();
            % obj.points.reset();
        end
        
        % function shuffle(obj)
        %     numObservations = obj.NumObservations;
        %     obj.data_indices = randperm(numObservations);
        % 
        %     obj.imagesA.Files = obj.imagesA.Files(obj.data_indices);
        %     obj.imagesA_cache = obj.imagesA_cache(obj.data_indices);
        %     obj.imagesB.Files = obj.imagesB.Files(obj.data_indices);
        %     obj.imagesB_cache = obj.imagesB_cache(obj.data_indices);
        %     % obj.points.Files = obj.points.Files(obj.data_indices);
        %     % obj.points_cache = obj.points_cache(obj.data_indices);
        %     obj.data_current_index = 1;
        % end
        function objNew = shuffle(obj)
            objNew = obj.copy();
            numObservations = objNew.NumObservations;
            objNew.imagesA = copy(obj.imagesA);
            objNew.points = copy(obj.points);
            objNew.data_indices = randperm(numObservations);

            objNew.imagesA.Files = objNew.imagesA.Files(objNew.data_indices);
            objNew.imagesB.Files = objNew.imagesB.Files(objNew.data_indices);
            objNew.points.Files = objNew.points.Files(objNew.data_indices);
            objNew.points_cache = objNew.points_cache(objNew.data_indices);
            objNew.imagesA_cache = objNew.imagesA_cache(objNew.data_indices);
            objNew.imagesB_cache = objNew.imagesB_cache(objNew.data_indices);
            obj.data_current_index=1;
        end
        
        function [aOut, bOut] = normaliseImages(obj, aIn, bIn)
            aOut = cellfun(@(x) single(x)/obj.ARange, aIn, 'UniformOutput', false);
            bOut = cellfun(@(x) single(x)/obj.BRange, bIn, 'UniformOutput', false);
        end
        
        function val = get.MiniBatchSize(obj)
            val = obj.MiniBatchSize_;
        end
        
        function set.MiniBatchSize(obj, val)
            obj.imagesA.ReadSize = val;
            obj.imagesB.ReadSize = val;
            obj.points.ReadSize = val;
            obj.MiniBatchSize_ = val;
        end
    end
end

function data = mat_reader(filename)
data = load(filename);
if isfield(data, 'img')
    % This is an image file
    data = single(data.img);
else
    % This is a point file
    data = single(data.cpts);
end
% data = single(data-700);
end

function [patchesA, patchesB] = sample_and_mask(imagesA, imagesB, cpts, patch_size)
% persistent sphere_mask;
% r = 5; % 5*1.5mm = 7.5 mm
% if isempty(sphere_mask)
%     rs = -r:r;
%     [xx,yy,zz]=meshgrid(rs,rs,rs);
%     d = sqrt(xx.^2 + yy.^2 + zz.^2);
%     sphere_mask = d<=r;
% end

N = length(imagesA);
patch_size_half = patch_size/2;
% b = (-patch_size_half+1):patch_size_half;
% c = -r:r;

patchesA = cell(size(imagesA));
patchesB = cell(size(imagesB));
for k=1:N
    imgA = imagesA{k};
    imgB = imagesB{k};
    pts = cpts{k};
    dim = size(imgA);
    p = pts(randi(size(pts,1)),:);
    y = p(1);
    x = p(2);
    z = p(3);
    
    y2 = max(y, patch_size_half(1)); y2 = min(y2, dim(1)-patch_size_half(1)); % keep safe
    x2 = max(x, patch_size_half(2)); x2 = min(x2, dim(2)-patch_size_half(2));
    z2 = max(z, patch_size_half(3)); z2 = min(z2, dim(3)-patch_size_half(3));

    ys = (y2-patch_size_half(1)+1) : (y2+patch_size_half(1));
    xs = (x2-patch_size_half(2)+1) : (x2+patch_size_half(2));
    zs = (z2-patch_size_half(3)+1) : (z2+patch_size_half(3));

    patch1a = imgA(ys, xs, zs);
    patch1b = imgB(ys, xs, zs);
    % patch1a = imgA(y-39:y+40,x-39:x+40, z-7:z+8);
    % patch1b = imgB(y-39:y+40,x-39:x+40, z-7:z+8);

    patch_mask = cast((patch1a== -5000), 'like', patch1a);
    patch1a(patch_mask==1) = 0;
    patch1a = cat(4, patch1a, patch_mask); % Patch 1a will have two channels: 1) the masked image, and 2) the mask

    patchesA{k} = patch1a;
    patchesB{k} = patch1b;
end
end

