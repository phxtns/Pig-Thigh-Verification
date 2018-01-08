function [ phantom ] = computePhantompressureinvolume( phantom, x, y, z, p_scale, ocl )
%COMPUTEPHANTOMPRESSUREINVOLUME Computes all pressure in volume
%calculations for children-style phantoms

% If there is no mask, we cannot perform the desired actions.
if (isfield(phantom, 'mask') && ~isequal(phantom.name{1}, 'Transducer'))
    
    Tx_center = phantom.center;
    Tx_ds     = phantom.ds;
    Tx_u_l    = phantom.vels_T_l;
    Tx_u_s    = phantom.vels_T_s;
    
    if ocl == 0
        disp('Occlusion testing is turned off.');
    elseif ocl == 1
        disp('Mesh-based occlusion testing is turned on.');
    elseif ocl == 2
        disp('Mask-based occlusion testing is turned on.');
    end
    
    
    if isfield(phantom, 'children')
        for i = 1:length(phantom.children)
            if isfield(phantom.children{i}, 'vels_R_l')
                Tx_center = cat(1, Tx_center, phantom.children{i}.center );
                Tx_ds     = cat(1, Tx_ds,     phantom.children{i}.ds     );
                Tx_u_l  = cat(1, Tx_u_l,  phantom.children{i}.vels_R_l);
                Tx_u_s  = cat(1, Tx_u_s,  phantom.children{i}.vels_R_s);
            end
        end
    end
    
    phantom.pressure_l = pressureinvolume(...
        phantom.k_l        ,... k
        phantom.params.c_l ,... c
        phantom.params.rho ,... rho
        Tx_center          ,... Tx_center
        Tx_ds              ,... Tx_ds
        Tx_u_l             ,... Tx_u
        x'                 ,... x
        y'                 ,... y
        z'                 ,... z
        phantom.mask       ,... mask
        int32(ocl)            ... ocl_mode
        );
    phantom.pressure_l = phantom.pressure_l.*sqrt(p_scale);

    if phantom.params.c_s > 0
        phantom.pressure_s = pressureinvolume(...
            phantom.k_s        ,... k
            phantom.params.c_s ,... c
            phantom.params.rho ,... rho
            Tx_center          ,... Tx_center
            Tx_ds              ,... Tx_ds
            Tx_u_s             ,... Tx_u
            x'                 ,... x
            y'                 ,... y
            z'                 ,... z
            phantom.mask       ,... mask
            int32(ocl)            ... ocl_mode
        );
        phantom.pressure_s = phantom.pressure_s.*sqrt(p_scale);
    end
end

clear Tx_center Tx_ds Tx_u_l Tx_u_s;

if isfield(phantom, 'children')
    for i = 1:length(phantom.children)
        phantom.children{i} = ...
            computePhantompressureinvolume( phantom.children{i}, x, y, z, p_scale, ocl );
    end
end

end

