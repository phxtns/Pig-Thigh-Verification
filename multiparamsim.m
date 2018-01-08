%% Code for running several simulations with varying parameters
%{
% Old parameter set
parameters = struct;
parameters.water = struct('c_l', 1486, 'c_s', 0, 'rho', 996, 'alpha_l', 2.88e-4, 'alpha_s', 0, 'kt', 0.615, 'ct', 4186, 'wt', 0);
parameters.pigfat = struct('c_l', 1426, 'c_s', 0, 'rho', 916, 'alpha_l', 16.6, 'alpha_s', 0, 'kt', 0.16, 'ct', 2.64e+03, 'wt', 4.25e-04);
parameters.water = struct('c_l', 1471, 'c_s', 0, 'rho', 978, 'alpha_l', 3.4, 'alpha_s', 0, 'kt', 0.615, 'ct', 4186, 'wt', 0);
parameters.pigmuscle = struct('c_l', 1580, 'c_s', 0, 'rho', 1041, 'alpha_l', 9, 'alpha_s', 0, 'kt', 0.47, 'ct', 3.72e3, 'wt', 0);
parameters.pigmuscle.wt = 0.027 * (parameters.pigmuscle.rho * 1e-6 * 1e3 / 60);
parameters.pigcorticalbone = struct('c_l', 3300, 'c_s', 0, 'rho', 1950, 'alpha_l', 152, 'alpha_s', 178, 'kt', 0.52, 'ct', 1.4e3, 'wt', 0);
parameters.pigcorticalbone.c_s = (1400/2550)*parameters.pigcorticalbone.c_l;
parameters.pigcorticalbone.wt = 0.14 * (parameters.pigcorticalbone.rho * 1e-6 * 1e3 / 60);
parameters.usephase = 0;
parameters.change = 'None';
parameters.tnum = 10;
parameters.multi = 0;
parameters.ocl = 0;

parameters = struct;
parameters.pigfat = struct('c_l', 1426, 'c_s', 0, 'rho', 916, 'alpha_l', 16.6, 'alpha_s', 0, 'kt', 0.16, 'ct', 2.64e+03, 'wt', 4.25e-04);
parameters.water = struct('c_l', 1465, 'c_s', 0, 'rho', 978, 'alpha_l', 1.5, 'alpha_s', 0, 'kt', 0.615, 'ct', 4186, 'wt', 0);
parameters.pigmuscle = struct('c_l', 1620, 'c_s', 0, 'rho', 1050, 'alpha_l', 3.6, 'alpha_s', 0, 'kt', 0.6, 'ct', 3.2e3, 'wt', 0);
parameters.pigmuscle.wt = 0.027 * (parameters.pigmuscle.rho * 1e-6 * 1e3 / 60);
parameters.pigcorticalbone = struct('c_l', 2400, 'c_s', 0, 'rho', 1700, 'alpha_l', 152, 'alpha_s', 266, 'kt', 0.5, 'ct', 1.1e3, 'wt', 0);
parameters.pigcorticalbone.c_s = 0.6*parameters.pigcorticalbone.c_l; %Originally (1400/2550)*
parameters.pigcorticalbone.wt = 0.14 * (parameters.pigcorticalbone.rho * 1e-6 * 1e3 / 60);
parameters.marrow = struct('c_l', 1620, 'c_s', 0, 'rho', 1050, 'alpha_l', 3.6, 'alpha_s', 0, 'kt', 0.6, 'ct', 3.2e3, 'wt', 0);
parameters.usephase = 0;
parameters.change = 'None';
parameters.tnum = 10;
parameters.multi = 0;
parameters.ocl = 0;



parameters = struct;
parameters.pigfat = struct('c_l', 1426, 'c_s', 0, 'rho', 916, 'alpha_l', 16.6, 'alpha_s', 0, 'kt', 0.16, 'ct', 2.64e+03, 'wt', 4.25e-04);
parameters.water = struct('c_l', 1465, 'c_s', 0, 'rho', 978, 'alpha_l', 1.5, 'alpha_s', 0, 'kt', 0.615, 'ct', 4186, 'wt', 0);
parameters.pigmuscle = struct('c_l', 1620, 'c_s', 0, 'rho', 1050, 'alpha_l', 3.6, 'alpha_s', 0, 'kt', 0.6, 'ct', 3.2e3, 'wt', 0);
parameters.pigmuscle.wt = 0.027 * (parameters.pigmuscle.rho * 1e-6 * 1e3 / 60);
parameters.pigcorticalbone = struct('c_l', 3500, 'c_s', 0, 'rho', 1910, 'alpha_l', 55, 'alpha_s', 86, 'kt', 0.32, 'ct', 1.31e3, 'wt', 0);
parameters.pigcorticalbone.c_s = 0.55*parameters.pigcorticalbone.c_l; %Originally (1400/2550)*
parameters.pigcorticalbone.wt = 0.14 * (parameters.pigcorticalbone.rho * 1e-6 * 1e3 / 60);
parameters.pigmarrow = struct('c_l', 1450, 'c_s', 0, 'rho', 1029, 'alpha_l', 12.5, 'alpha_s', 0, 'kt', 0.28, 'ct', 2.67e3, 'wt', 0);
parameters.usephase = 0;
parameters.change = 'None';
parameters.tnum = 10;
parameters.multi = 0;
parameters.ocl = 0;

%}
parameters = struct;
parameters.pigfat = struct('c_l', 1426, 'c_s', 0, 'rho', 916, 'alpha_l', 16.6, 'alpha_s', 0, 'kt', 0.16, 'ct', 2.64e+03, 'wt', 4.25e-04);
parameters.water = struct('c_l', 1470, 'c_s', 0, 'rho', 978, 'alpha_l', 2.3, 'alpha_s', 0, 'kt', 0.615, 'ct', 4186, 'wt', 0);
parameters.pigmuscle = struct('c_l', 1579, 'c_s', 0, 'rho', 1090, 'alpha_l', 3.2, 'alpha_s', 0, 'kt', 0.68, 'ct', 3e3, 'wt', 0);
parameters.pigmuscle.wt = 0.027 * (parameters.pigmuscle.rho * 1e-6 * 1e3 / 60);
parameters.pigcorticalbone = struct('c_l', 2850, 'c_s', 0, 'rho', 1900, 'alpha_l', 97, 'alpha_s', 105, 'kt', 0.6, 'ct', 1.3e3, 'wt', 0);
parameters.pigcorticalbone.c_s = 0.55*parameters.pigcorticalbone.c_l; %Originally (1400/2550)*
parameters.pigcorticalbone.wt = 0.01 * (parameters.pigcorticalbone.rho * 1e-6 * 1e3 / 60);
parameters.pigmarrow = struct('c_l', 1410, 'c_s', 0, 'rho', 1029, 'alpha_l', 5.1, 'alpha_s', 0, 'kt', 0.28, 'ct', 2.67e3, 'wt', 0);
parameters.pigmarrow.wt = 0.14 * (parameters.pigmarrow.rho * 1e-6 * 1e3 / 60);
parameters.usephase = 0;
parameters.change = 'None';
parameters.tnum = 10;
parameters.multi = 0;
parameters.ocl = 0;
parameters.t_base = 33;
parameters.coolrate = 2/(12*60); % Rate T_base drops as K per second.

%% New Parameter Set

currparameters = parameters;
currparameters.usephase = 0;
currparameters.tnum = 10;
%currparameters.focus = [-0.0495,0.092,0.0138];
currparameters.change = 'oclonadjmusclec';
currparameters.ocl = 2;
simulate(currparameters);

currparameters = parameters;
currparameters.usephase = 0;
currparameters.tnum = 11;
%currparameters.focus = [-0.0505,0.098,0.0138];
currparameters.t_base = 31.5;
currparameters.change = 'oclonadjmusclec';
currparameters.ocl = 2;
simulate(currparameters);

currparameters = parameters;
currparameters.usephase = 0;
currparameters.tnum = 12;
currparameters.t_base = 29.5;
currparameters.coolrate = 1/(12*60);
%currparameters.focus = [-0.0495,0.104,0.0138];
currparameters.change = 'oclonadjmusclec';
currparameters.ocl = 2;
simulate(currparameters);

currparameters = parameters;
currparameters.usephase = 0;
currparameters.tnum = 13;
currparameters.t_base = 29;
currparameters.coolrate = 1/(12*60);
%currparameters.focus = [-0.0495,0.11,0.0138];
currparameters.change = 'oclonadjmusclec';
currparameters.ocl = 2;
simulate(currparameters);

currparameters = parameters;
currparameters.usephase = 0;
currparameters.tnum = 14;
currparameters.t_base = 29;
currparameters.coolrate = 1/(12*60);
currparameters.change = 'oclonadjmusclec';
currparameters.ocl = 2;
simulate(currparameters);
%{
currparameters = parameters;
currparameters.usephase = 0;
currparameters.tnum = 15;
currparameters.t_base = 27.5;
currparameters.change = 'oclonadj';
currparameters.ocl = 2;
simulate(currparameters);

currparameters = parameters;
currparameters.usephase = 0;
currparameters.tnum = 16;
currparameters.t_base = 27;
currparameters.change = 'oclonadj';
currparameters.ocl = 2;
simulate(currparameters);
%}
%{
currparameters = parameters;
currparameters.multi = 1;
currparameters.usephase = 1;
currparameters.focus = [-0.049,0.09,0.0138];
currparameters.pigcorticalbone.c_s = 0.5*currparameters.pigcorticalbone.c_l;
currparameters.change = 'New Parameter Set RevRay Used c_s0.5c_l';
simulate(currparameters);

currparameters = parameters;
currparameters.multi = 1;
currparameters.usephase = 1;
currparameters.focus = [-0.049,0.09,0.0138];
currparameters.pigcorticalbone.c_s = 0.6*currparameters.pigcorticalbone.c_l;
currparameters.change = 'New Parameter Set RevRay Used c_s0.6c_l';
simulate(currparameters);

currparameters = parameters;
currparameters.multi = 1;
currparameters.usephase = 1;
currparameters.focus = [-0.049,0.09,0.0138];
currparameters.pigcorticalbone.c_s = 0.7*currparameters.pigcorticalbone.c_l;
currparameters.change = 'New Parameter Set RevRay Used c_s0.7c_l';
simulate(currparameters);
%}
