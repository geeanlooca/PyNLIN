clc; clear all;

%% fiber index profile
dco = 17e-6;    % core diameter [m]
rco=dco / 2;	% core radius [m]
nco=1.461;      % core refractive index
ncl=1.46;       % cladding refractive index
Delta = 0.5 * 1e-2;
nco = ncl / sqrt(1-Delta)

% numerical aperture
NA = sqrt(nco^2-ncl^2);

% normalized frequency
lambda = 1450e-9;
V = 2*pi * rco ./ lambda * NA;
V = 5;
lambda = 2*pi * rco ./ V * NA;

% working wavelength [nm]
% lambda=1550;    % for the time being we consider only one frenquency

% considered modes
themodes='lp';
% fiber creation
sif=SIF('nco', nco, 'ncl', ncl, 'radius', rco, ...
    'wavelength', lambda);
[cof, modes]=sif.cutoffFreq('lp');

fprintf('Number of modes: %d, counting degeneracies: %d\n', modes.nmodes, ...
    modes.ndegmodes);
v = sif.normCutoffFreq();

%%
if ~exist('b', 'var')
    b = sif.normPropagationConst(modes)';
end
