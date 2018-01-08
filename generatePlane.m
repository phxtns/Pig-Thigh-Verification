function generatePlane(MRIfol, mods)
% This is a code for generating the planes of interest from simulation
% result and saving them to a the folder Planes of Interest.

segmentedPhantomFile = 'Segmenting/Pig 25/segmentation.mat'; % Location of segmentation file
result = [MRIfol, '_', mods]; %Simulation result filename in Results/
T_base = 33;

dicoms = dir(['MRI Thermometry/' MRIfol '/*.dcm']);
hdr = dicominfo(['MRI Thermometry/' MRIfol '/' dicoms(1).name]); % MRI thermometry dicom info

% Loop through all the MRI thermometry pixels and get their coordinates
ys = 1:hdr.Rows;
xs = 1:hdr.Columns;
[x_mesh, ~] = meshgrid(xs,ys);

coords = zeros(size(x_mesh,1),size(x_mesh,2),3);

for i = 1:numel(x_mesh)
    [R, C] = ind2sub(size(x_mesh), i);
    MRpos = MRpixelCoordinate(hdr, C, R);
    coords(R,C,:) = MRpos(1:3)*1e-3;
end

temp_figure = figure('visible', 'off');

disp(sprintf('Loading necessary files...'))
load([segmentedPhantomFile]);
load(['Results/' result '.mat'], 'T_end');

simulated = cell(size(T_end));
for i = 1:length(T_end)
    disp(sprintf('Processing slide %d of %d', i, size(T_end,1)))
    T_curr = T_end{i} - T_base;
    %{
    %% Set bone voxels to 0, keeping neighbours to non-bone
    for i1 = 2:size(T_curr,1)-1
        for i2 = 2:size(T_curr,2)-1
            for i3 = 2:size(T_curr,3)-1
                if phantom.mask(i1,i2,i3) == 0
                    if ~((phantom.mask(i1-1,i2,i3) == 1) || (phantom.mask(i1+1,i2,i3) == 1) || ...
                            (phantom.mask(i1,i2-1,i3) == 1) || (phantom.mask(i1,i2+1,i3) == 1) || ...
                            (phantom.mask(i1,i2,i3-1) == 1) || (phantom.mask(i1,i2,i3+1) == 1))
                        T_curr(i1,i2,i3) = 0;
                    end
                end
            end
        end
    end
    %}
    %T_curr(phantom.children{1}.mask) = 0;
    %T_curr(phantom.children{2}.mask) = 0;
    planeOfInterest = slice(x,y,z,T_curr,coords(:,:,1),coords(:,:,2),coords(:,:,3));
    
    simulated{i} = planeOfInterest.CData;
end

close(temp_figure);

filename = ['Compare/Planes Of Interest/Slices_' result '.mat'];
save(filename, 'simulated', 'coords');
end