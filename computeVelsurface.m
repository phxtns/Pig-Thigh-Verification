function [ phantom ] = computeVelsurface( phantom, freq, x, y, z, multitrans, ocl)
%COMPUTEVELSURFACE Computes all velsurface calculations for children-style
%phantoms.

%% Wave Numbers

% The wave numbers are established for the initial phantom.
phantom.k_l = 2*pi*freq / phantom.params.c_l - 1j * phantom.params.alpha_l * freq/1e6; % Longitudinal complex wave number
if phantom.params.c_s > 0
    phantom.k_s = 2*pi*freq / phantom.params.c_s - 1j * phantom.params.alpha_s * freq/1e6; % Shear complex wave number
else
    phantom.k_s = 0; % Shear waves are not supported, i.e. material is a fluid
end

% Compute internal reflections for all media after the transducer.
if multitrans && (~(isequal(phantom.name{1}, 'Transducer') || isequal(phantom.name{1}, 'Test Transducer')  || isequal(phantom.name{1}, 'Muscle') || isequal(phantom.name{1}, 'Bone') || isequal(phantom.name{1}, 'Marrow')))
    count = 0;
    [phantom,count] = computeInternals(phantom, freq, x, y, z, count);
    disp(sprintf('There were %d internal reflections calculated.', count));
end

if isfield(phantom, 'children')
    for i = 1:length(phantom.children)
        disp(sprintf('Propagating from %s to %s...', phantom.name{1}, phantom.children{i}.name{1}));
        [phantom.children{i}.vels_T_l,phantom.children{i}.vels_R_l, ...
            phantom.children{i}.vels_T_s, phantom.children{i}.vels_R_s] = ...
            velsurface(...
            phantom.k_l                    ,... k1_l
            phantom.k_s                    ,... k1_s
            phantom.params.c_l             ,... c1_l
            phantom.params.c_s             ,... c1_s
            phantom.params.rho             ,... rho1
            phantom.center                 ,... Tx_center
            phantom.normal                 ,... Tx_norm
            phantom.ds                     ,... Tx_ds
            phantom.vels_T_l               ,... vels_l_x
            phantom.vels_T_s               ,... vels_s_x
            phantom.children{i}.params.c_l ,... c2_l
            phantom.children{i}.params.c_s ,... c2_s
            phantom.children{i}.params.rho ,... rho2
            phantom.children{i}.center     ,... Sx_center
            phantom.children{i}.normal     ,... Sx_norm
            int32(ocl)                     ,... ocl_mode
            phantom.mask                   ,... mask
            x'                             ,... x
            y'                             ,... y
            z'                              ... z
        );
    
        % Call computeVelsurface for all children.
        [phantom.children{i}] = computeVelsurface(phantom.children{i}, freq, x, y, z, multitrans, ocl);

    end
end

end

