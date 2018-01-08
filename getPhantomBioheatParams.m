function [ bioheatparams ] = getPhantomBioheatParams( bioheatparams, phantom, x, y, z, freq, T_base)
%Getphantombioheatparams Returns a structure with all parameters necessary
%for running bioheat simulations given the input children-style phantom
parameters = Parameters();
if ~isfield(bioheatparams, 'Domain')
    %first time being run
    bioheatparams.Domain = int32(-1*ones(length(y),length(x),length(z)));
    bioheatparams.kappa = [];
    bioheatparams.Ct = [];
    bioheatparams.rho = [];
    bioheatparams.Ct_b = [];
    bioheatparams.rho_b = [];
    bioheatparams.W_b = [];
    bioheatparams.T_b = [];

    bioheatparams.Qr = zeros(length(y),length(x),length(z));
end

d_i = length(bioheatparams.kappa);

if ((isfield(phantom, 'mask')) && ~isequal(phantom.name{1}, 'Transducer'))
    blood = parameters.blood;
    bioheatparams.Domain(phantom.mask) = d_i;
    bioheatparams.kappa = cat(1, bioheatparams.kappa, phantom.params.kt);
    bioheatparams.Ct = cat(1, bioheatparams.Ct, phantom.params.ct);
    bioheatparams.rho = cat(1, bioheatparams.rho, phantom.params.rho);
    bioheatparams.Ct_b = cat(1, bioheatparams.Ct_b, blood.ct);
    bioheatparams.rho_b = cat(1, bioheatparams.rho_b, blood.rho);
    bioheatparams.W_b = cat(1, bioheatparams.W_b, phantom.params.wt);
    bioheatparams.T_b = cat(1, bioheatparams.T_b, T_base);

    Qr = (abs(phantom.pressure_l).^2)*((phantom.params.alpha_l*(freq/1e6)) / (phantom.params.rho*phantom.params.c_l));
    if phantom.params.c_s > 0
        Qr = Qr + (abs(phantom.pressure_s).^2)*((phantom.params.alpha_s*(freq/1e6)) / (phantom.params.rho*phantom.params.c_s));
    end

    bioheatparams.Qr(phantom.mask) = Qr(phantom.mask); clear Qr;
end

if isfield(phantom, 'children')
    for i = 1:length(phantom.children)
        bioheatparams = getPhantomBioheatParams(...
            bioheatparams, phantom.children{i}, x, y, z, freq, T_base);
    end
end

end