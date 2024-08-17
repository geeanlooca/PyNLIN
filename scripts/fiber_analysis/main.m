%% MAIN RAMAN FILE
% Demonstrates simple multi-signal multi-pump fiber Raman amplifier
% This file is used as a reference program

%% Amplifier parameters
amp_length = 100e3; % [m]

maxpumps = 5;
maxchannels = 50;
minpower = 10;
maxpower = 500;
minlambda = 1450;
maxlambda = 1510;
%

pumpPower = 50e-3;
pumpLambda = [1450]*1e-9;
% pumpPower = (minpower + (maxpower-minpower).*rand(npumps,1)) *1e-3;
% pumpLambda = (mpinlambda + (maxlambda-minlambda).*rand(npumps,1)) * 1e-9;

% pumpPower = [376 273 195 126 56 44 26 16] *1e-3; % [W]
% pumpLambda = [1407.2 1421.3 1436.4 1452.6 1466.1 1495.5 1495.8 1496.7] * 1e-9; %[m]


Nch = 1;
% Nch = randi(maxchannels);
WDMcenter = 1550e-9; 
spacing = 50e9;

signalLambda = WDM.comb(Nch, WDMcenter, spacing, 'units', 'lambda'); % [m]
signalPower = 5e-6 * ones(size(signalLambda)); % [W]

alpha_s = convert.alpha2linear(0.2);
alpha_p = convert.alpha2linear(0.2);

dz = 20;

opt = struct;
opt.pump_direction = 1; %(sign(randn(npumps, 1)));
% opt.rkstepper = @rk4step;
opt.undepleted_pump = false;
opt.sigsigint = false;
opt.debug = true;
opt.error_threshold = 5e-4;
opt.mex = false;

algorithms = { 'interior-point', 'sqp' };

%% Solve the Raman equations
tic;
[z, sig, pump, output] = raman_solve_rk4(signalPower, pumpPower, signalLambda, ...
    pumpLambda, alpha_s, alpha_p, amp_length, dz, opt);
toc;

sig = squeeze(sig);
pump = squeeze(pump);

tic;
[z, sigoff, pumpoff, output] = raman_solve_rk4(signalPower, 0*pumpPower, signalLambda, ...
    pumpLambda, alpha_s, alpha_p, amp_length, dz, opt);
toc;

onoffGain = 10*log10(sig ./ squeeze(sigoff));

%% Plot the results
figure(7);
clf;
hold on;
plot(z * 1e-3, convert.watt2dBm(sig));
plot(z * 1e-3, convert.watt2dBm(pump));
plot(z * 1e-3, convert.watt2dBm(sigoff), '--');
% plot(convert.watt2dBm(sig));
% plot(convert.watt2dBm(pump));
hold off;
xlabel('Length [km]');
ylabel('Power [dBm]');
title('Signal evolution');
grid on;
grid minor;

% 
    figure(3);
    f=clf;
    semilogy(signalLambda * 1e9, onoffGain(end, :));
    xlabel('Length [km]');
    ylabel('Gain [dB]');
%

% Plot the spectrum
lambda = [pumpLambda(:); signalLambda(:)];
figure(15);
clf;
for i = 1 : 1
    powers = [pump(end, :) sig(end, :)]';
    bar(lambda * 1e9, convert.watt2dBm(powers));
    xlabel('$\lambda [nm]$');
    ylabel('Power [dBm]');
    title('Spectrum at fiber end');
end

return;

%%


