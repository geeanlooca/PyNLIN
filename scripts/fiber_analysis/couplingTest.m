%% RAMAN AMPLIFICATION WITH MODAL EFFECTS - v0.1
% Script for simulating the Raman amplification equations with multiple
% pumps and multiple signals at different wavelengths and different LP
% modes. The considered model is the one presented in [1].
%
% [1] - Ryf, R., Essiambre, R., von Hoyningen-Huene, J., & Winzer,  P. 
%       (2012). Analysis of Mode-Dependent Gain in Raman Amplified Few-Mode
%       Fiber. Optical Fiber Communication Conference, (3), OW1D.2. 
%       https://doi.org/10.1364/OFC.2012.OW1D.2
clear all;

%% TODO: Fix the SIF class calculating the overlap integrals
% This is currently NOT WORKING
%% ========================================================================

%% Fiber description
fprintf(['Defining the fiber and computing cutoff frequencies, ', ...
    'propagation constants, and overlap integrals\n']);

dco = 17e-6;    % core diameter [m]
rco=dco / 2;	% core radius [m]
ncl=1.46;       % cladding refractive index
Delta = 0.5 * 1e-2;
nco = ncl / sqrt(1-Delta); % core refractive index
nc0 = 1.4673;

% Create the object containing the description of the fiber used
fiber = StepIndexFiber(nco, ncl, rco);

% numerical aperture
V = 5;
lambda = 2*pi * rco ./ V * fiber.NA;
fiber.wavelength = [1450 1500 1550 1560]*1e-9;

% force computation of overlap integrals
OI = fiber.getOverlapIntegrals;
nmodes = fiber.modeCount;

%% Amplifier parameters
ampLength = 50e3; % [m]

% % pumpPower = [273 245 370 215] * 1e-3; % [W]
% pumpLambda = [1420 1430 1450 1465]*1e-9; % [m]
pumpPower = 250e-3;
pumpLambda = 1450e-9;

Npu = length(pumpLambda);

Nch = 1;
WDMcenter = 1545e-9;
spacing = 100e9;

signalLambda = WDM.comb(Nch, WDMcenter, spacing, 'units', 'lambda'); % [m]
% signalLambda = 1560e-9;
signalPower = convert.dBm2watt(-20) * ones(size(signalLambda)); % [W]

alpha_s = convert.alpha2linear(0.2);
alpha_p = convert.alpha2linear(0.35);

% Integration step
dz = 20;

opt = struct;
opt.pump_direction = -1; 
opt.undepleted_pump = false;
opt.sigsigint = false;
opt.debug = true;
opt.error_threshold = 1e-3;
opt.mex = false;
opt.fiber = fiber;


%% Solve the Raman equations
fprintf('Solving the Raman propagation equations');
tic;
[z, sig, pump, output] = raman_solve_rk4(signalPower, pumpPower, signalLambda, ...
    pumpLambda, alpha_s, alpha_p, ampLength, dz, opt);
toc;
sig = reshape(sig, [], nmodes, Nch);
pump = reshape(pump, [], nmodes, Npu);

%% Plot the results
styles = { '-', '--', '-.', ':' };
labels = fiber.modes.string;
handles = zeros(nmodes, 1);
figure(2);
clf;
hold on;    
    for m = 1:nmodes
        plot(z * 1e-3, convert.watt2dBm(squeeze(sig(:, m, :))), ...
            'Color', 'k','LineStyle', styles{m});
        plot(z * 1e-3, convert.watt2dBm(squeeze(pump(:, m, :))), 'LineStyle', styles{m});
        handles(m) = plot(NaN,NaN,'Color', 'k', 'LineStyle', styles{m});
    end
hold off;
xlabel('Length [km]');
ylabel('Power [dBm]');
title('Signal evolution');
grid on;
grid minor;
legend(handles, labels);


%% Plot the on-off gain at the end of the fiber
figure(5);
clf;
hold on;
signalOffPower= convert.watt2dBm(signalPower / nmodes * exp(-ampLength * alpha_s));
for m = 1:nmodes
    
    sigEndPower = convert.watt2dBm(squeeze(sig(end, m, :)));
    onoffGain = sigEndPower - signalOffPower;
    
    plot(signalLambda * 1e9, onoffGain, ...
        'LineStyle', styles{m}, 'Marker', 'x', 'DisplayName', labels{m});    
%     handles(m) = plot(NaN,NaN,'Color', 'k', 'LineStyle', styles{m});
end
hold off;
xlabel('$\lambda [nm]$');
ylabel('On-off Gain [dB]');
l=legend('show');
set(l, 'Location', 'northwest');
title('Power spectrum');
grid on;

return;

%% Plot the spectrum
lambda = [pumpLambda(:); signalLambda(:)];
figure(15);
hold on;
clf;
for i = 1 : nmodes
    p = squeeze(pump(end, i, :));
    s = squeeze(sig(end, i, :));
    powers = [p(:) ; s(:) ]';
    bar(convert.watt2dBm(powers));
    xlabel('$\lambda [nm]$');
    ylabel('Power [dBm]');
    title('Spectrum at fiber end');
end