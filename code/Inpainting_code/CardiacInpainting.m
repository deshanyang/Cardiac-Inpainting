FolderLocation = which('CardiacInpainting.m');
%A is the artifact impacted image, B is ground truth, Points is sampling
%for patch extraction
TestInpaintA = [FolderLocation(1:end-19) 'TSImageInpaintTestA'];
TestInpaintB = [FolderLocation(1:end-19) 'TSImageInpaintTestB'];
TestPointsA= [FolderLocation(1:end-19) 'TSImageInpaintTestApoints'];
if ~isfolder(TestInpaintA) mkdir(TestInpaintA); end
if ~isfolder(TestInpaintB) mkdir(TestInpaintB); end
if ~isfolder(TestPointsA) mkdir(TestPointsA); end
%% 
checkpoint_folder = [FolderLocation(1:end-19) 'Checkpoints'];
if ~isfolder(checkpoint_folder) mkdir(checkpoint_folder); end
Nslice = 16;
patch_size = [80 80 Nslice];
range = 1000;
learn_rate = 1e-4;
resume_from = '';
options = p2p.trainingOptions('InputChannels', 2, 'OutputChannels', 1, 'MiniBatchSize', 20, 'ARange', range, 'BRange', range, 'MaxEpochs', 100, ...
    'GDepth', 4, 'DDepth', 4, 'PreSize', [], 'InputSize', patch_size, 'CheckpointPath', checkpoint_folder, 'ResumeFrom', resume_from, ...
    'DLearnRate', learn_rate, 'GLearnRate', learn_rate, 'Lambda', 20, 'VerboseFrequency', 20);

%% 
%Used for training of a new model with different datapoints
% p2pModel = p2p.p2ptrain_cardiac(TrainInpaintA, TrainInpaintB, TrainPointsA, ValidateInpaintA, ValidateInpaintB, ValidatePointsA, options);

%% Loading in the pretrained model
load([FolderLocation(1:end-19) 'p2p_checkpoint_2024-04-12T23-04-22_0082.mat'])

%% Testing of the trained model
files = dir(TestInpaintA);
filesb = dir(TestInpaintB);
samplepoints = dir(TestPointsA);
NF = length(files);
TestCases = cell(1,length(files)-2);
a=fspecial3("gaussian",[80,80,32],40);
a=a/max(a,[],'all');
SaveLocation = [FolderLocation(1:end-19) 'InpaintingResults'];
if ~isfolder(SaveLocation) mkdir(SaveLocation); end
for idx = 3:NF
    load(fullfile(filesb(idx).folder,filesb(idx).name));
    exampleTarget = img;
    load(fullfile(files(idx).folder,files(idx).name));
    exampleMask = img;
    exampleMask = exampleMask<-2000;
    load(fullfile(samplepoints(idx).folder,samplepoints(idx).name));
    inpainted=zeros(size(img));
    weights = zeros(size(img));
    filename = files(idx).name;
    filename = filename(1:18);
    predictions = cell(0);
    prediction_count = 0;
    for patch=1:length(cpts)
        ys = cpts(patch,1)-39:cpts(patch,1)+40;
        xs = cpts(patch,2)-39:cpts(patch,2)+40;
        zs = cpts(patch,3)-15:cpts(patch,3)+16;
        exampleInput1 = img(ys,xs,zs);
        maskInput = cast(exampleInput1 == -5000,'like',exampleInput1);
        exampleInput1(maskInput>0)=0;
        exampleInput=cat(4,exampleInput1,maskInput);
        weights(ys,xs,zs)=weights(ys,xs,zs)+a;
        exampleOutput = p2p.translate(p2pModel,exampleInput,'ARange',range);
        exampleOutput(maskInput==0) = exampleInput1(maskInput==0);
        if any(maskInput(:)>0)
            prediction_count = prediction_count+1;
            predictions{prediction_count} = cat(2,exampleInput1,exampleOutput);
        end
        inpainted(ys,xs,zs)=inpainted(ys,xs,zs)+exampleOutput.*a;
    end
    inpainted = inpainted./weights;
    inpainted(exampleMask==0)=img(exampleMask==0);
    inpainted(isnan(inpainted))=exampleTarget(isnan(inpainted));
    errmap=(inpainted/1000)-(double(exampleTarget)/1000);
    L1 = abs(errmap.^2);
    L1a=L1(exampleMask==1);
    if any(L1a(:)>1e-1)
        TestL1(idx-2) = mean(L1a(L1a>1e-1), 'all','omitnan') + mean(L1a, 'all','omitnan');
    else
        TestL1(idx-2) = 2*mean(L1a,'all','omitnan');
    end
    % v3d(cat(2,img,inpainted,exampleTarget),[1.5 1.5 1.5]);
    
    TestCases{1,idx-2}=cat(2,img,inpainted,exampleTarget);
end
save(fullfile(SaveLocation,'TestCases'),"TestCases")

%% Inference on your own data:
%patient dir of cases to test
imglist=dir('');
%Directory of artifact masks that are correlated to the images
masklist = dir('');
load(fullfile(imglist(1).folder,imglist(1).name));
inpainted4d=zeros([size(img), 10]);
a=fspecial3("gaussian",patch_size,40);
a=a/max(a,[],'all');
for phase=1:length(imglist)
    load(fullfile(imglist(phase).folder,imglist(phase).name));
    img=double(img)-1000; %only necessary if original data is 0:4096 not -1000:3096
    imgorig=img;
    % Load in the associated artifact mask;
    load(fullfile(masklist(phase).folder,masklist(phase).name));
    artifacts=(artifacts>0);
    artifacts = imdilate(artifacts,strel('disk',3,4));
    cpts = GetPatientSamplePoints(img,artifacts,patch_size);
    inpainted=zeros(size(img));
    weights = zeros(size(img));
    Out4d = [];
    Art4d = [];
    predictions = cell(0);
    prediction_count = 0;
    for patch=1:length(cpts)
        ys = cpts(patch,1)-39:cpts(patch,1)+40;
        xs = cpts(patch,2)-39:cpts(patch,2)+40;
        zs = cpts(patch,3)-15:cpts(patch,3)+16;
        exampleInput1 = img(ys,xs,zs);
        maskInput = artifacts(ys,xs,zs);
        exampleInput1(maskInput>0)=0;
        exampleInput=cat(4,exampleInput1,maskInput);
        weights(ys,xs,zs)=weights(ys,xs,zs)+a;
        exampleOutput = p2p.translate(p2pModel,exampleInput,'ARange',range);
        Out4d(:,:,:,patch)=exampleOutput;
        exampleOutput(maskInput==0) = exampleInput1(maskInput==0);
        Art4d(:,:,:,patch)=exampleOutput;
        if any(maskInput(:)>0)
            prediction_count = prediction_count+1;
            predictions{prediction_count} = cat(2,exampleInput1,exampleOutput);
        end
        inpainted(ys,xs,zs)=inpainted(ys,xs,zs)+exampleOutput.*a;
    end
    inpainted = inpainted./weights;
    inpainted(~artifacts)=img(~artifacts);
    zerofill = img;
    zerofill(artifacts)=0;
    inpainted4d(:,:,:,phase)=inpainted;
end

