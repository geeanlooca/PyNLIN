classdef RamanResponse
    %RAMANRESPONSE Class defining models for representing the Raman vibrational response in silica fibers.
    %
    %   Collection of methods for obtaining/approximating the Raman
    %   vibrational response of silica optical fibers. The models
    %   implemented in this class are taken from:
    %
    %       [1] Hollenbeck, D., & Cantrell, C. D. (2002).
    %           Multiple-vibrational-mode model for fiber-optic Raman gain
    %           spectrum and response function. J Opt Soc Am B, 19(12),
    %           2886–2892. https://doi.org/10.1364/JOSAB.19.002886
    %
    %   See also RAMANRESPONSE.IMPULSERESPONSE
    
    %% Properties
    properties (Constant)
        position = [ 56.25 100.00 231.25 362.50 463.00 497.00 611.50 ...
            691.67 793.67 835.50 930.00 1080.00 1215.00 ] * 1e2; % [m]
        
        intensity = [ 1.00 11.40 36.67 67.67 74.00 4.50 6.80 4.60 4.20 ...
            4.50 2.70 3.10 3.00 ];
        
        gFWHM = [ 52.10 110.42 175.00 162.50 135.33 24.50 41.50 ...
            155.00 59.50 64.30 150.00 91.00 160.00 ] * 1e2;
        
        lFWHM = [ 17.37 38.81 58.33 54.17 45.11 8.17 13.83 51.67 ...
            19.83 21.43 50.00 30.33 53.33 ] * 1e2;
        
        omega_v = 2*pi*constants.c0 * ...
            RamanResponse.position;  % Vibrational frequencies
        
        gamma = pi * constants.c0 * ...
            RamanResponse.lFWHM;     % Lorentzian FWHM
        
        Gamma = pi * constants.c0 * ...
            RamanResponse.gFWHM;     % Gaussian FWHM
        
        A = RamanResponse.intensity .* ...
            RamanResponse.omega_v;  % Amplitudes
        
        Ncomp = length(RamanResponse.position); % number of vibrational components
    end
    
    %% Methods
    methods (Static)
        function res = impulseResponse(varargin)
            %IMPULSERESPONSE Model the vibrational Raman impulse response
            %in silica optical fibers.
            %
            %   res = IMPULSERESPONSE(duration, samples) implements the
            %   raman impulse response using the 'intermediate broadening'
            %   model [1] (default) where 'samples' is the numer of samples
            %   used, and 'duration' is the total duration (in seconds) of
            %   the impulse response.
            %
            %   res = IMPULSERESPONSE(duration, samples, model). As
            %   above, but can specify a different model from the default.
            %
            %   model can be one of the following:
            %       - 'ib': intermediate broadening [1] (default)
            %
            %   res = IMPULSERESPONSE(duration, samples, model, Name,
            %   Value). As above, but can specify Name-Value parameter
            %   pairs.
            %
            %   Supported parameters:
            %       - 'normalize': logical scalar (default false)            %
            %
            %   References:
            %       [1] Hollenbeck, D., & Cantrell, C. D. (2002).
            %           Multiple-vibrational-mode model for fiber-optic Raman gain
            %           spectrum and response function. J Opt Soc Am B, 19(12),
            %           2886–2892. https://doi.org/10.1364/JOSAB.19.002886
            %
            %   See also RAMANRESPONSE.IMPULSERESPONSE
            
            
            p = inputParser;
            
            %validModels = {'ib', 'pibo', 'phbo'};
            validModels = {'ib', 'default'};
            
            p.addRequired('duration', @(x) validateattributes(x, ...
                {'double'}, {'scalar', 'positive'}));
            
            p.addRequired('samples', @(x) validateattributes(x, ...
                {'numeric'}, {'scalar', 'integer', 'positive'}));
            
            p.addOptional('model', 'ib', @(x) validateattributes(x, ...
                {'char'}, {'scalartext'}));
            
            p.addParameter('normalize', false, @(x) validateattributes(x, ...
                {'logical'}, {'scalar'}));
                        
            
            % Parse the inputs
            parse(p, varargin{:});
            
            % Validate selected model
            if ~any(strcmpi(validModels, p.Results.model))
                error(['Model not supported. See the help for the list' ...
                    ' supported models']);
            end
            
            % Implentation
            T = p.Results.duration;
            Ntime = p.Results.samples;
            t = linspace(0, T, Ntime);
            
            res = zeros(size(t));
            
            A = RamanResponse.A;
            gamma = RamanResponse.gamma;
            Gamma = RamanResponse.Gamma;
            omegav = RamanResponse.omega_v;
            Ncomp = length(omegav);
            
            switch lower(p.Results.model)
                
                % Intermediate Broadening Model [1], Sec. 3
                case {'ib', 'default'}
                    
                    % add each vibrational model to the impulse response
                    for l = 1 : Ncomp
                        coeff = A(l) / (omegav(l));
                        factor = coeff * exp(-gamma(l) * t) ...
                            .* exp(-Gamma(l)^2 * t.^2/4) .* ...
                            sin(omegav(l) * t);
                        
                        res = res  + factor;
                    end
                    
                otherwise
                    error('Unsupported model');
                    
            end
            
            % normalize to 1 if required
            if p.Results.normalize
                res = res / abs(max(res));
            end
        end
        
        function [res, f] = frequencyResponse(varargin)
            %FREQUENCYRESPONSE Compute the Raman vibrational frequency
            %response, from which the Raman gain spectrum can be obtained.
            
            defaultUnit = 'Hz';
            
            p = inputParser;
            
            p.addRequired('bandwidth', @(x) validateattributes(x, ...
                {'double', 'single'}, {'scalar', 'positive'}));
            
            p.addRequired('spacing', @(x) validateattributes(x, ...
                {'double', 'single'}, {'scalar', 'positive'}));
            
            p.addOptional('model', 'ib', @(x) validateattributes(x, ...
                {'char'}, {'scalartext'}));
            
            p.addParameter('normalize', true, @(x) validateattributes(x, ...
                {'logical'}, {'scalar'}));
            
            p.addParameter('units', defaultUnit, @(x) validateattributes(x, ...
                {'char'}, {'scalartext'}));
            
            parse(p, varargin{:});
                        
            if p.Results.spacing > p.Results.bandwidth
                error('Frequency spacing cannot be greater than the observed bandwidth');
            end
            
            % check for scaling values
            acceptedUnits = {'THz', 'GHz', 'Hz', 'default'};
            if ~any(strcmpi(acceptedUnits, p.Results.units) ~=1)
                error('Units not specified');
            end
            
            scaling = 1;
            switch lower(p.Results.units)
                case {'hz', defaultUnit}
                    scaling = 1;
                case {'ghz'}
                    scaling = 1e9;
                case {'thz'}
                    scaling = 1e12;
            end
            
            spacing = p.Results.spacing * scaling;
            bandwidth = p.Results.bandwidth * scaling; 
            
            % Get the Raman impulse response
            impRespSamples = round(bandwidth * 2 / spacing);
            impRespDuration = 1/spacing;
            
            impResp = RamanResponse.impulseResponse(impRespDuration, ...
                impRespSamples, p.Results.model);
            
            % Obtain the frequency response by Fourier-transforming
            freqResp = fftshift(fft(impResp));
            
            % Built the frequency axis
            f = linspace(-bandwidth, bandwidth, impRespSamples);
            f = f / scaling;
            
            res = freqResp;
        end
        
        function [gains, gainSpectrum, f] = gain(varargin)
            %GAIN Computes the raman gain at the specified frequency shift.
                        
            defaultUnit = 'Hz';
            
            p = inputParser;            
            
            p.addRequired('shift', @(x) validateattributes(x, ...,
                {'double', 'single'}, {'real'}));
            
            p.addRequired('bandwidth', @(x) validateattributes(x, ...
                {'double', 'single'}, {'scalar', 'positive'}));
            
            p.addRequired('spacing', @(x) validateattributes(x, ...
                {'double', 'single'}, {'scalar', 'positive'}));
            
            p.addOptional('model', 'ib', @(x) validateattributes(x, ...
                {'char'}, {'scalartext'}));            
            
            p.addParameter('normalize', true, @(x) validateattributes(x, ...
                {'logical'}, {'scalar'}));
            
            p.addParameter('units', defaultUnit, @(x) validateattributes(x, ...
                {'char'}, {'scalartext'}));
            
            parse(p, varargin{:});
            
            % validate input
            if max(abs(p.Results.shift)) > p.Results.bandwidth
                error('Observation bandwidth is too small for specified frequency shift');
            end            
            
            % obtain the frequency response
            freqRespFunc = @RamanResponse.frequencyResponse;
            [freqResp, f] = freqRespFunc(p.Results.bandwidth, ...
                p.Results.spacing, p.Results.model, ...
                'units', p.Results.units);

            
            % Raman gain is proportional to the imaginary part of the
            % frequency response
            gainSpectrum = imag(freqResp);
            
            % normalize if needed            
            if p.Results.normalize
                gainSpectrum = gainSpectrum / abs(max(gainSpectrum));
            end
            
            % get the data points
            gains = spline(f, gainSpectrum, p.Results.shift);
        end
    end
end

