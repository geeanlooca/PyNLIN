%% Amplifier parameters

amp_length = 10e3; % [m]

pumpPower = 350e-3;
pumpLambda = [1450]*1e-9;
% pumpPower = (minpower + (maxpower-minpower).*rand(npumps,1)) *1e-3;
% pumpLambda = (minlambda + (maxlambda-minlambda).*rand(npumps,1)) * 1e-9;

% pumpPower = [376 273 195 126 56 44 26 16] *1e-3; % [W]
% pumpLambda = [1407.2 1421.3 1436.4 1452.6 1466.1 1495.5 1495.8 1496.7] * 1e-9; %[m]


% LP groups: LP01, LP11, LP21, LP02
nmodes = 4;

overlapInt = [6.24 4.12 2.85 4.62; 4.12 4.36 3.81 2.33; 2.85 3.81 3.88 2.12; 4.62 2.33 2.12 6.15];


Nch = 80;
% Nch = randi(maxchannels);
WDMcenter = 1550e-9; 
spacing = 200e9;

signalLambda = WDM.comb(Nch, WDMcenter, spacing, 'units', 'lambda'); % [m]
signalPower = 1e-4 * ones(size(signalLambda)); % [W]

alpha_s = convert.alpha2linear(0.2); %convert.alpha2linear(0.2);
alpha_p = convert.alpha2linear(0.35); % convert.alpha2linear(linspace(0.27, 0.35, length(pumpPower)));
    
dz = 20;

opt = struct;
opt.pump_direction = 1; %(sign(randn(npumps, 1)));
% opt.rkstepper = @rk4step;
opt.undepleted_pump = false;
opt.sigsigint = false;
opt.debug = false;
opt.error_threshold = 5e-4;
opt.mex = false;
opt.nmodes = 4;

algorithms = { 'interior-point', 'sqp' };

%% Solve the Raman equations
tic;
[z, sig, pump, output] = raman_solve_rk4(signalPower, pumpPower, signalLambda, ...
    pumpLambda, alpha_s, alpha_p, amp_length, dz, opt);
toc;
% [z, sigoff, pumpoff] = raman_solve_rk4(signalPower, 0*pumpPower, signalLambda, ...
%     pumpLambda, alpha_s, alpha_p, amp_length, opt);

% compute the on-off Raman gain
% GainOnOff = sig(end, :) ./ sigoff(end, :);
% 

%% Plot the results
figure(7);
clf;
hold on;
plot(z * 1e-3, convert.watt2dBm(sig));
plot(z * 1e-3, convert.watt2dBm(pump));
% plot(convert.watt2dBm(sig));
% plot(convert.watt2dBm(pump));
hold off;
xlabel('Length [km]');
ylabel('Power [dBm]');
title('Signal evolution');
grid on;
grid minor;

% 
if false
    figure(3);
    f=clf;
    semilogy(z * 1e-3, abs(sig' - spline(zmod, sigmod', z)));
    xlabel('Length [km]');
    ylabel('Error');
end
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

figure(3);
clf;
hold on;
% plot(signalLambda * 1e9, convert.watt2dB(GainOnOff));
hold off;
xlabel('\lambda [nm]');
ylabel('Gain [dB]');
title('On-off gain');
grid on;
grid minor;
ylim([0 20]);


