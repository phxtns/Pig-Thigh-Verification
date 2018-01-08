function simulate(params)

% User inputs
pigNumber = 25; %pig number
disp(sprintf('Running simulation for Pig %d.', pigNumber))
treatmentNumber = params.tnum; %obtained from Excel sheet for simplicity
disp(sprintf('Treatment number is %d.', treatmentNumber))
T_base = params.t_base; %ºC
disp(sprintf('Base temperature is %g degrees Celsius.', T_base))

freq = 0.518e6; %Hz
disp(sprintf('Frequency is %dHz.', freq))
dt = 5e-3; %temporal discretization [s] %Typically 5e-3

%% Files and Folders
segmentedPhantomFile = sprintf('Segmenting/Pig %d/segmentation.mat', pigNumber);
treatmentsFile = sprintf('Treatments/pig%d.mat', pigNumber);

%% Paths
addpath('Arrays', genpath('../Thermometry Simulation Code'), 'MRI Thermometry', 'Parameters', 'Compare');

%% Load treatment info and phantom geometry
disp('Loading segmentation and treatment files...')
segm = load(segmentedPhantomFile);
phantom = segm.phantom;
x = segm.x;
y = segm.y;
z = segm.z;
dh = segm.dh;
clear 'segm';

treatments = load(treatmentsFile);
treatments = treatments.treatments;
treatment = treatments(find([treatments.num] == treatmentNumber));

thermFol = sprintf('E%dS%03d', treatment.dicomExam, treatment.dicomSeries); %folder where MRI thermometry dicoms are stored

%% Parse some geometry info
focus = (treatment.mri*1e-3)'; %focal spot of sonication
Affine = treatment.th.transform; %get affine transform from treament info
shift = 1e-2 * Affine(1:3,4)'; %transducer centre (originally in cm)
U = Affine(1:3, 1:3); %transducer rotation matrix                                                                                   

%% Get transducer geometry and set as first phantom element
disp('Acquiring transducer geometry and shifting to MRI space...')
water = params.water;
lambda = water.c_l/freq;
Tx = getArray([-0.005 0 0], lambda/6);
for i = 1:size(Tx.center,1) %rotate each element
    Tx.center(i,:) = U*Tx.center(i,:)' + shift';
    Tx.normal(i,:) = U*Tx.normal(i,:)';
end

Tx.mask = logical(ones(size(phantom.mask))) & ~phantom.mask & ~phantom.children{1}.mask & ~phantom.children{2}.mask;
Tx.name = {'Transducer'};

Tx.children = {phantom};
phantom = Tx;
clear 'Tx';

%% Set phantom parameters
disp('Establishing phantom parameters...')
phantom.params = water; %transducer layer
pigmuscle = params.pigmuscle;
phantom.children{1}.params = pigmuscle; %muscle layer
phantom.children{1}.parentparams = water; %muscle layer
pigcortbone = params.pigcorticalbone; %bone layer
phantom.children{1}.children{1}.params = pigcortbone; %anterior bone layer
phantom.children{1}.children{1}.parentparams = pigmuscle; %anterior bone layer
phantom.children{1}.children{2}.params = pigcortbone; %posterior bone layer
phantom.children{1}.children{2}.parentparams = pigmuscle; %posterior bone layer
pigmarrow = params.pigmarrow;
phantom.children{1}.children{1}.children{1}.params = pigmarrow; %posterior bone layer
phantom.children{1}.children{1}.children{1}.parentparams = pigcortbone; %posterior bone layer

%% Calculate the initial velocities (assuming 1m/s - see line 63) according to phase data
disp('Calculating initial velocities and deactivating inactive elements...')
phantom.vels_T_l = complex(zeros(size(phantom.ds)));

if params.usephase == 1
    disp('Generating phase data using reverse ray tracing...')
    velocityvolume = ones(size(phantom.mask));
    velocityvolume(phantom.mask) = phantom.params.c_l;
    velocityvolume(phantom.children{1}.mask) = phantom.children{1}.params.c_l;
    velocityvolume(phantom.children{1}.children{1}.mask) = phantom.children{1}.children{1}.params.c_l;
    velocityvolume(phantom.children{1}.children{2}.mask) = phantom.children{1}.children{2}.params.c_l;
    velocityvolume(phantom.children{1}.children{1}.children{1}.mask) = phantom.children{1}.children{1}.children{1}.params.c_l;

    step = 0.001;
    focus = params.focus;
    for i = 1:size(phantom.center,1)
        dvec = step*(phantom.center(i,:) - focus)/norm(phantom.center(i,:) - focus);
        avvels = 0;
        for j = 1:floor(norm(phantom.center(i,:) - focus)/step)-1
            curpos = focus + j*dvec;
            [a xind] = min(abs(curpos(:,1) - x));
            [a yind] = min(abs(curpos(:,2) - y));
            [a zind] = min(abs(curpos(:,3) - z));
            avvels = avvels + velocityvolume(yind,xind,zind);
        end
        avvels = avvels/(floor(norm(phantom.center(i,:) - focus)/step) - 1);
        phantom.avvels(i,1) = avvels;
    end

    phaseData = [];
    for i = 1:length(phantom.avvels)
        phaseData(i,1) = exp(1j*(2*pi*freq/phantom.avvels(i))*norm(phantom.center(i,:) - focus));
    end
        
    save(['Results/' thermFol '_' params.change '_PhaseData.mat'], 'phaseData');
    phantom.vels_T_l = phaseData;
else
    disp('Loading phase data from treatment files...')
    phaseData = treatment.phasedata;
    for i = 1:size(phantom.subelems, 1) %assign same phase data to all subelements of same element
        phantom.vels_T_l(phantom.subelems(i,1):phantom.subelems(i,2)) = 1 * exp(1j * phaseData(i));
    end
end

for i = 1:size(phantom.inactives, 1) %remove inactive elements
    phantom.vels_T_l(phantom.subelems(phantom.inactives(i),1):phantom.subelems(phantom.inactives(i),2)) = 0;
end

phantom.vels_T_s = []; %no shear waves

%% STEP 1: Propagate velocity at surface boundaries
disp('Initialising velocity propagator...')
phantom = computeVelsurface(phantom, freq, x, y, z, params.multi, params.ocl);
%save('Tempfile.mat', 'phantom');

%% STEP 2: Calculate the pressure in the volumes while sonicating
disp('Initialising pressure calculator...')
P = treatment.power; %transducer array power in W
p_scale = P / ( sum(phantom.ds) * phantom.params.rho * phantom.params.c_l * 0.5); %calculate scaling factor due to 1m/s assumption
phantom = computePhantompressureinvolume(phantom, x, y, z, p_scale, params.ocl);
%save('Tempfile.mat', 'phantom');

%% STEP 3: Calculate the heating after treatment
% Obtain details from MRI Thermometry in order to compare correctly
disp('Initialising bioheat solver...')

list = dir(['MRI Thermometry/' thermFol '/*.dcm']);
info1 = dicominfo(['MRI Thermometry/' thermFol '/' list(1).name]);
info4 = dicominfo(['MRI Thermometry/' thermFol '/' list(4).name]);
bioheatStep = (info4.TriggerTime - info1.TriggerTime) / 1000;
infoE = dicominfo(['MRI Thermometry/' thermFol '/' list(end-2).name]);
bioheatEnd = (infoE.TriggerTime - info1.TriggerTime) / 1000;

if params.tnum ~= 16
    nextthermFol = sprintf('E%dS%03d', treatment.dicomExam, treatment.dicomSeries + 1); %folder where MRI thermometry dicoms are stored
    infocurr = dicominfo(['MRI Thermometry/' thermFol '/' list(end-2).name]);
    list2 = dir(['MRI Thermometry/' nextthermFol '/*.dcm']);
    infonext = dicominfo(['MRI Thermometry/' nextthermFol '/' list2(end-2).name]);
    
    currtime = [str2num(infocurr.SeriesTime(1:2)),str2num(infocurr.SeriesTime(3:4)),str2num(infocurr.SeriesTime(5:6))];
    nexttime = [str2num(infonext.SeriesTime(1:2)),str2num(infonext.SeriesTime(3:4)),str2num(infonext.SeriesTime(5:6))];
    
    bioheatEnd2 = sum([3600,60,1].*(nexttime-currtime));
else
    bioheatEnd2 = bioheatEnd;
end

% Obtain bioheat parameters from children-style phantom
bp = getPhantomBioheatParams(struct(), phantom, x, y, z, freq, T_base);

%Generate time-dependent Q(t) = Qr*Qt(t)
duration = treatment.actualDuration; %treatment duration in sec
Qt = ones(length(0:dt:duration), 1); %using 100% duty cycle, Qt is always 1
preOff = ceil(bioheatStep/dt); %time steps before sonication is turned on
postOff = ceil((bioheatEnd2 - duration - bioheatStep)/dt); %time steps after sonication is turned off
Qt = cat(1, zeros(preOff, 1), Qt, zeros(postOff, 1)); %add pre and post steps

% Calculate Temperature at each bioheat step
bioheatSteps = 0:bioheatStep:bioheatEnd;
if bioheatEnd2 ~= bioheatEnd
    bioheatSteps(end+1) = bioheatEnd2;
end
discretizationSteps = 0:dt:bioheatEnd2;
T_end = cell(length(bioheatSteps), 1); %will hold temperature distribution at each step
if params.tnum == 10
    T_end{1} = T_base*ones(length(y),length(x),length(z)); %inital temperature distribution
else
    lastthermfol = sprintf('E%dS%03d', treatment.dicomExam, treatment.dicomSeries - 1); %folder where MRI thermometry dicoms are stored
    lastfile = load(['Results/' lastthermfol '_' params.change '.mat'], 'T_end');
    lastfile = lastfile.T_end;
    T_end{1} = lastfile{end};
end

for i = 2:length(bioheatSteps)
    Qt_partial = Qt(bioheatSteps(i-1) <= discretizationSteps & discretizationSteps < bioheatSteps(i));
    Nt = length(Qt_partial);
    bp.T_b = bp.T_b - params.coolrate*dt*Nt;
    T_end{i} = bioheat(bp.Domain, bp.Qr, Qt_partial, dh, dt, Nt, bp.kappa, bp.Ct, bp.rho, ...
        bp.Ct_b, bp.rho_b, bp.W_b, bp.T_b, T_end{i-1});
end

%% Save results
filename = ['Results/' thermFol '_' params.change '.mat'];
disp(sprintf('Saving data to %s.', filename))
save(filename, 'phantom', 'T_end', 'focus', 'water', 'pigmuscle', 'pigcortbone', '-v7.3');

%% Generate Thermometry Planes
generatePlane(thermFol,params.change);

clearvars -except parameters;
%reset(gpuDevice);
%delete('Tempfile.mat');
end