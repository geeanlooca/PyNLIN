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

nmodes = size(b, 2);
freq = convert.lambda2freq(lambda);
%% Compute the mode field distribution for each propagating mode
modeProfiles = cell(modes.ndegmodes);

pts = 2^10;
k = 3;
x = linspace(-rco * k, rco * k, pts);
y = linspace(-rco * k, rco * k, pts);

modeIndex = 1;
for i = 1 : modes.nmodes
    
    m = modes(i);
    ndeg = m.ndegmodes;
    type = m.labels{m.modes(1)};
    
    l = m.modes(2);
    p = m.modes(3);
    
    fprintf('LP%d%d. Degenerations: %d\n', l, p, m.ndegmodes);
    
    propconst = b(i);
    
    chicore = V * sqrt(1-propconst) / rco;
    chiclad = V * sqrt(propconst) / rco;
    
    [X, Y] = ndgrid(x, y);
    [THETA, R] = cart2pol(X, Y);
    phasemaskCOS = cos(l * THETA);
    phasemaskSIN = sin(l * THETA);
    
    radialmask = getRadialMask(l, R, chicore, chiclad, rco);
    
    intensityProfile = zeros([size(radialmask) 2]);
    intensityProfile(:, :, 1) = radialmask .* phasemaskCOS;
    
    intensityProfile(:, :, 2) = radialmask .* phasemaskSIN;
    if l == 0
        intensityProfile(:, :, 2) = intensityProfile(:, :, 1);
    end
    
    
    for c = 0 : ndeg/2 - 1
        modeProfiles{modeIndex+c} = intensityProfile(:, :, 1);
        modeProfiles{modeIndex+c+ndeg/2} = intensityProfile(:, :, 2);
    end
    
    modeIndex = modeIndex + ndeg;
end

%% Drawing
mode = 10;

% Plot the selected mode
figure(2);
% imagesc(x/rco, y/rco, modeProfiles{mode});
surf(x/rco, y/rco, abs(modeProfiles{mode}));
shading interp;
grid on;
grid minor;

% draw the fiber
c = [0 0];
pos = [c-rco 2*rco 2*rco];
rectangle('Position', pos, 'Curvature', [1 1], 'FaceColor', [0.5 0.5 0.5, 0.3]);
axis equal tight;
xlabel('r/r_{co}');
ylabel('r/r_{co}');


% Compute the effective mode area of the fiber as the inverse of the
% overlap integral of the LP01 (fundamental) mode
LP01 = modeProfiles{1};
Aeff = 1/OverlapIntegral(LP01, LP01, x, y) * 1e12;

fprintf('True core area: %.3f um^2\n', pi * (rco*1e6)^2);
fprintf('Effective core area: %.3f um^2\n', Aeff);

%% Compute the overlap integrals between each mode
% 
disp('Compute the overlap integrals between each mode...');
ovIntegrals = zeros(modes.ndegmodes);
for i = 1 : modes.ndegmodes
    for j = 1 : modes.ndegmodes
        ovIntegrals(i, j) = OverlapIntegral(modeProfiles{i}, ...
            modeProfiles{j}, x, y) * 1e-9;
    end
end

%% Plot the mode profiles
disp('Plot the mode profiles');
figure(3);
clf;
cols = 5;
rows = ceil(modes.ndegmodes / cols);
for i = 1 : modes.ndegmodes
    subplot(rows, cols, i);
    imagesc(x, y, modeProfiles{i});
    c = [0 0];
    pos = [c-rco 2*rco 2*rco];
    rectangle('Position', pos, 'Curvature', [1 1], 'FaceColor', [0.5 0.5 0.5, 0.3]);
    
    axis equal tight;
end

function E = getRadialMask(l, r, chico, chicl, a)
    core_idx = abs(r) <= a;
    rcore = r(core_idx);
    clad_idx = abs(r) > a;
    rclad = r(clad_idx);
    E = zeros(size(r));
    Ecore = besselj(l, chico * rcore) ./ besselj(l, chico * a);
    Eclad = besselk(l, chicl * rclad) ./ besselk(l, chicl * a);
    E(core_idx) = Ecore;
    E(clad_idx) = Eclad;
end