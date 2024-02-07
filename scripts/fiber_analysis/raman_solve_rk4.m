function [zSolved, signal, pump, output] = raman_solve_rk4( varargin )
%RAMAN_SOLVE_RK4 Solve the Raman amplifier equations in a single-mode fiber
%   [z, signal, pump] = RAMAN_SOLVE_RK4(signal_power, pump_power,
%   signal_wavelength, pump_wavelength, length, options)
%
%   Solve the Raman propagation equation for a set of pumps and signals.
%   signal_power and pump_power arguments are vectors of the same size of
%   signal_wavelength and pump_wavelength, respectively. They can also be
%   scalars; in this case the supplied signal/pump power will be applied
%   to all the signals/pumps specified by the
%   signal_wavelength/pump_wavelength vector.
%
%   Parameters
%   ----------
%
%   pump_direction: vector of the same size of pump_powers. Positive value 
%   at position j in the vector indicates that the j-th pump is 
%   co-propagating with the signals, while negative values indicate 
%   counterpropagating pumps. Can also be a scalar; in this case the value 
%   will be applied to every pump. Default: 1.
%
%   undepleted_pump: set to true if the undepleted pump approximation has to 
%   be applied. This is done by setting to 0 the appropriate elements in the 
%   Raman gain matrix. Default: false.
%
%   sigsigint: set to false if the signal-signal interaction has to be 
%   neglected. Notice that the pump-pump interaction is still considered. 
%
%   Returns
%   -------
%
%   z: Nx1 real vector
%   
%   signal: NxS real vector
%
%   pump: NxP real vector
%
%See also RK4, RK4STEP, RAMAN_SOLVE, ODE45

%% Default parameter values
defaultDirection = 1; % 1 for co-pumping, -1 for counterpumping
defaultError = 1e-3; % determines the stopping condition in case of counterp.

%% Parsing scheme
parser = inputParser;

% Must be set to true so I can pass additional arguments to the functions
% called by raman_solve.
parser.KeepUnmatched = true;

% Function name to display in case of errors
parser.FunctionName = 'raman_solve_rk4';

validator = @(x) validateattributes(x, {'numeric'}, {'real', 'nonnegative', ...
    'vector', 'finite', 'nonnan'});

% Required parameters
parser.addRequired('signal_power', validator);
parser.addRequired('pump_power', validator);
parser.addRequired('signal_wavelength', validator);
parser.addRequired('pump_wavelength', validator);
parser.addRequired('alpha_s', @(x) validateattributes(x, ...
    {'numeric'}, {'real', 'vector', 'finite', 'nonnegative'}));
parser.addRequired('alpha_p', @(x) validateattributes(x, ...
    {'numeric'}, {'real', 'vector', 'finite', 'nonnegative'}));
parser.addRequired('length', @(x) validateattributes(x, ...
    {'numeric'}, {'real', 'scalar', 'finite', 'positive'}));
parser.addRequired('dz', @(x) validateattributes(x, {'numeric'}, ...
    {'scalar', 'positive', 'real'}));

% Parameters
parser.addParameter('pump_direction', defaultDirection, ...
    @(x) validateattributes(x, {'numeric'}, {'real', 'vector'}));

parser.addParameter('error', defaultError, @(x) validateattributes(x, ...
    {'numeric'}, {'scalar', 'positive', 'finite'}));

parser.addParameter('debug', false, @(x) validateattributes(x, ...
    {'logical'}, {'scalar'}));

% Set flag to consider the undepleted pump approximation
% The signals do not affect the pump
parser.addParameter('undepleted_pump', false, @(x) validateattributes(x, ...
    {'logical'}, {'scalar'}));

% Set this flag to false if you want to ignore signal-signal interaction
parser.addParameter('sigsigint', true, @(x) validateattributes(x, ...
    {'logical'}, {'scalar'}));
parser.addParameter('pumppumpint', true, @(x) validateattributes(x, ...
    {'logical'}, {'scalar'}));

parser.addParameter('mex', false, @(x) validateattributes(x, ...
    {'logical'}, {'scalar'}));

parser.addParameter('algorithm', 'interior-point');

parser.addParameter('error_threshold', 1e-4);

% Modal parameters

% Number of modes 
parser.addParameter('nmodes', 1, @(x) validateattributes(x, ...
    {'numeric'},{'integer', 'real', 'scalar', '>=', 1}));

% Fraction of power of each mode
parser.addParameter('pumpPowerFraction', []);
parser.addParameter('signalPowerFraction', []);

parser.addParameter('overlapIntegrals', [], @(x) validateattributes(x, ...
    {'numeric'}, {'real', 'nonnegative'}));

%% Parse the arguments
parse(parser, varargin{:});

%% Error checking on the input parameters
res = parser.Results;

% Check if number of wavelengths and number of pump powers is compatible
if isscalar(res.signal_power)
    res.signal_power = ones(size(res.signal_wavelength)) * res.signal_power;
end

if isscalar(res.pump_power)
    res.pump_power = ones(size(res.pump_wavelength)) * res.pump_power;
end

if isscalar(res.pump_direction)
    res.pump_direction = ones(size(res.pump_wavelength)) * res.pump_direction;
end

if any(size(res.signal_power) ~= size(res.signal_wavelength))
    error('Signal and signal wavelength vectors must have the same dimension');
end

if any(size(res.pump_power) ~= size(res.pump_wavelength))
    error('Signal and signal wavelength vectors must have the same dimension');
end

% Pump direction
if nnz(res.pump_direction) ~= numel(res.pump_direction)
    error(['Pump direction cannot be zero. It must be either'
        ' a positive or negative number']);
end

%% TODO: think of a way to select the fraction of power on each mode for the
% pumps


% Number of modes
nmodes = res.nmodes;

% Signals and pump powers are specified for each wavelength: the power in
% each mode is computed from these values using the 'pumpPowerFraction' and
% 'signalPowerFraction' parameters
signalCount = length(res.signal_wavelength);
pumpCount = length(res.pump_wavelength);
totalSig = (signalCount + pumpCount);


% Check input power fractions determining the amount of power in each mode
% for each wavelength
if ~isempty(res.pumpPowerFraction)
    if length(res.pumpPowerFraction) == nmodes
        
        if sum(res.pumpPowerFraction) ~= 1
            error('Pump power fraction must sum to 1');            
        end        
        
        % has to be repeated 'pumpCount' times
        pumpPowerFraction = repmat(res.pumpPowerFraction(:), pumpCount, 1);
        
    elseif length(res.pumpPowerFraction) == nmodes * pumpCount
        
        for i=1:pumpCount
            if sum(res.pumpPowerFraction((i-1)*nmodes + 1, i*nmodes)) ~= 1
                error('Pump power fraction must sum to 1 for each wavelength');
            end
        end
        
        pumpPowerFraction = res.pumpPowerFraction;
        
    else
        error('pumpPowerFraction size not compatible with the specified number of modes');
    end
else
    % if not specified, split the power equally among modes
    pumpPowerFraction = repmat(ones(nmodes, 1)/nmodes, pumpCount, 1);
end

if ~isempty(res.signalPowerFraction)
    if length(res.signalPowerFraction) == nmodes
        
        if sum(res.signalPowerFraction) ~= 1
            error('Signal power fraction must sum to 1');            
        end        
        
        % has to be repeated 'signalCount' times
        signalPowerFraction = repmat(res.signalPowerFraction(:), signalCount, 1);
        
    elseif length(res.signalPowerFraction) == nmodes * pumpCount
        
        for i=1:pumpCount
            if sum(res.signalPowerFraction((i-1)*nmodes + 1, i*nmodes)) ~= 1
                error('Signal power fraction must sum to 1 for each wavelength');
            end
        end
        
        signalPowerFraction = res.signalPowerFraction;
        
    else
        error('signalPowerFraction size not compatible with the specified number of modes');
    end
else
    % if not specified, split the power equally among modes
    signalPowerFraction = repmat(ones(nmodes, 1)/nmodes, signalCount, 1);
end
%--------------------------------------------------------------------------


% Function that repeats the values by nmodes
reshize = @(x) reshape(repmat(x, nmodes, 1), [], 1);

pumpPower = reshize(res.pump_power(:)) .* pumpPowerFraction;
sigPower = reshize(res.signal_power(:)) .* signalPowerFraction;


% Pump direction is transformed to a vector of -1 and +1 (for counter and
% co-pumping, respectively
pumpDir = reshize(sign(res.pump_direction));

if any(size(res.pump_power) ~= size(res.pump_direction))
    % throw an error if the size of the arrays are not compatible    
    error('Pump power and pump direction vector must be of the same size');
end

% Determine the pumping configuration, i.e. co-propagating,
% counter-propagating or bi-directional. Call the appropriate algorithm
% based on this parameter.

configuration = [];
if all(pumpDir == 1)
    % co-propagating    
    configuration = 0;
    
elseif all(pumpDir == -1)
    % counter-propagating
    configuration = 1;
else
    % bi-directional
    configuration = 2;
end

% Determine if there are any counterpropagating pumps
counterpumping = configuration > 0;
if counterpumping
    % disable counterpumping as it is not currently working correctly
    % warning(['Counterpropagating pumps are still being debugged.']);
end

%--------------------------------------------------------------------------
% Align the signal powers, wavelengths and attenuation in a single column
% vector. The first N values of these vectors are related to the signals,
% and the last M to the pumps.
%--------------------------------------------------------------------------

lambda_p = res.pump_wavelength;
lambda_s = res.signal_wavelength;
lambda_t = [lambda_s(:); lambda_p(:)];

%
alpha_p = res.alpha_p(:) .* ones(pumpCount * nmodes, 1);
alpha_s = res.alpha_s * ones(signalCount * nmodes, 1);
alpha = [alpha_s(:); alpha_p(:)];

%% Obtain the Raman gain spectrum
% Convert wavelengths to frequency
freq = convert.lambda2freq(lambda_t);

% Parameters for determining the Raman response
F = max(freq) * 10;
spacing = 1e10;

% Compute the frequency shifts between each signal
shifts = zeros(totalSig); 
for s = 1 : totalSig
    shifts(s, :) = freq(s) - freq;
end

% get raman gains
[gains, gainSpectrum, fx] = RamanResponse.gain(shifts, F, spacing);
if parser.Results.debug
    f = figure(99);
    f.Name = 'Raman response';
    plot(fx * 1e-12, -gainSpectrum);
    xlim([0 50]);
    xlabel('$\Delta$ f [THz]');
    ylabel('Im[h(t)]');
end

% % TODO: gain matrix is not symmetric here????
gains = triu(gains) - triu(gains, 1)';


% set diagonal equal to 1 => there is no "self" Raman interaction
% TODO: how to extend this to the multimodal case? Do modes in the same
% wavelength interact with each other through SRS?
gains = gains-diag(diag(gains));


% Set some coefficients to zero if the undepleted pump flag is set to true
% or if signal-signal interaction must be neglected
if res.undepleted_pump == true
    gains(signalCount+1:end, 1:signalCount) = 0;
end

if res.sigsigint == false
    gains(1:signalCount, 1:signalCount) = 0;
end

if res.pumppumpint == false
    gains(signalCount+1:end, signalCount+1:end) = 0;
end

% Use the Kronecker product to repeat columns and rows of the matrix a
% number of times equal to the number of modes
gains = kron(gains, ones(nmodes));


gainConstant = RamanConstants.typicalGain;

if ~isempty(res.overlapIntegrals)
    
    if size(res.overlapIntegrals, 1) == nmodes && size(res.overlapIntegrals, 2) == nmodes        
        % repeat this block matrix for each pump and signal wavelength
        OI = repmat(res.overlapIntegrals, totalSig, totalSig);
        
    elseif size(res.overlapIntegrals, 1) == nmodes * totalSig && ...
            size(res.overlapIntegrals, 2) == nmodes * totalSig
        % Do nothing
        OI = res.overlapIntegrals;
    else
        error("Unsupported size for parameter 'overlapIntegrals'");
    end
    
    ramanGain = gains * gainConstant .* OI;
else
    
    if nmodes > 1
        warning("Multimode fiber specified, but 'overlapIntegrals' param. not set");
    end
    
    % Default parameters for a generic SMF fiber
    Aeff = 80 * (1e-6)^2;
    ramanGain = gains * gainConstant / Aeff;
end

% output structure holding additional data used
out = struct();
out.raman_gain = ramanGain;

%% Prepare the solver
solver = struct();
if res.mex
    solver.function = @raman_ode_mex;
else
    solver.function = @raman_ode_opt;
end

solver.dz = res.dz;  % integration step
solver.interval = [0 res.length]; % integration interval
% initial condition
solver.y0 = [sigPower(:) ; pumpPower(:)];

% Prepare the structure containing the parameters of the system to pass to
% the function describing the system of equations
param = struct();
param.signals = signalCount * nmodes;
param.pumps = pumpCount * nmodes;
param.freqs = freq;

% TODO: check if this is correct
F = max(1, freq * (1./freq)'); % f(i) / f(j)
% F = ones(size(freq));
% F(F<1) = 1;
param.F = kron(F, ones(nmodes));
param.gain = ramanGain;

% keyboard;

param.G = param.F .* param.gain;
param.alpha = alpha;
param.direction = [ones(signalCount*nmodes, 1); pumpDir(:)];
param.P0 = solver.y0;
param.dz = solver.dz;
param.length = res.length;

% Prepare the structure to hold output information
output = struct();

%% Solve the system of equations using a fourth-order Runge-Kutta method
elapsed = [];
switch configuration
    case 0
        % Co-pumping: simple Runge-Kutta method
        tic;
        [zSolved, Psolved] = rk4(@(t, y) raman_ode_opt(t, y, param), ...
            solver.interval, solver.dz, solver.y0, ...
            parser.Unmatched); 
        elapsed = toc;
    case {1,2}
        % Counterpumping: shooting is needed
        tic;        
        [zSolved, Psolved, optout] = optimshooting(solver, param, ...
            'debug', parser.Results.debug, ...
            'algorithm', parser.Results.algorithm, ...
                'threshold', parser.Results.error_threshold);
        
        output.optim = optout;
        elapsed = toc;
end
output.time = elapsed;


%% return the results
zSolved = zSolved(:);
signal = Psolved(:, 1:signalCount*nmodes);

signal = reshape(signal, size(signal, 1), nmodes, signalCount);

pump = Psolved(:, signalCount*nmodes + 1:end);
pump=reshape(pump, size(pump, 1), nmodes, pumpCount);
    
end
