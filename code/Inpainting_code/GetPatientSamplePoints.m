function [centerpoints] = GetPatientSamplePoints(image,artifacts,patchsize)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
imgsize=size(image);
artifact = artifacts>0;
halfpatch = patchsize/2;
ys=floor(linspace(halfpatch(1),imgsize(1)-halfpatch(1),imgsize(1)/40));
xs=floor(linspace(halfpatch(2),imgsize(2)-halfpatch(2),imgsize(2)/40));
zs=floor(linspace(halfpatch(3),imgsize(3)-halfpatch(3),imgsize(3)/8));
cpts=[];

for y=1:length(ys)
    for x=1:length(xs)
        for z=1:length(zs)
            if sum(artifact(ys(y)-39:ys(y)+40,xs(x)-39:xs(x)+40,zs(z)-7:zs(z)+8),'all')>0
                cpts = cat(1,cpts,[ys(y),xs(x),zs(z)]);
            end
        end
    end
end
centerpoints=cpts;

end