classdef WDM
    %WDM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Property1
    end
    
    methods (Static)
        
        function plotComb(comb, power)
            %PLOTCOMB Plots the spectrum of a WDM comb
            
        end
        
        function wdm = comb(varargin)
            %COMB Get a list of wavelengths/frequencies of a WDM comb.
            %
            %   wdm = COMB(Nch, f0, df) creates an array of
            %   Nch frequencies centered at f0, spaced by df.
            %
            %   wdm = COMB(Nch, f0, df, 'units', value) as above, but if
            %   'units' is set to 'lambda' interprets f0 as a central
            %   wavelenght and df as the frequency spacing. The units of
            %   measurents must be compatible, e.g. nanometers and
            %   gigahertz. In this case the resulting array consists of
            %   wavelenght values centered at wavelength f0 and uniformly
            %   spaced in frequency by df.
            
            defaultUnit = 'lambda';
            validUnits = {'lambda', 'wavelength', 'frequency'};
            
            p = inputParser;
            
            % Set parsing scheme            
            p.addRequired('channels', @(x) validateattributes(x, ...
                {'numeric'}, {'integer', 'scalar', 'positive', 'finite'}));
            
            p.addRequired('center', @(x) validateattributes(x, ...
                {'numeric'}, {'real', 'scalar', 'positive', 'finite'}));
            
            p.addRequired('spacing', @(x) validateattributes(x, ...
                {'numeric'}, {'real', 'scalar', 'positive', 'finite'}));
            
            p.addParameter('units', defaultUnit, @(x) validateattributes(x, ...
                {'char'}, {'scalartext'}));
            
            parse(p, varargin{:});
            
            % Validate units
            if ~any(strcmpi(p.Results.units, validUnits))
                error('Units not valid');
            end
                        
            
            Nch = p.Results.channels;            
            rescale = false;
            
            switch p.Results.units
                case 'frequency'
                    
                    f0 = p.Results.center;
                    df = p.Results.spacing;
                    
                case {'lambda', 'wavelength'}
                    % first convert to frequency                    
                    f0 = convert.lambda2freq(p.Results.center);
                    df = p.Results.spacing;
                    rescale = true;
            end
                
            
            % create WDM comb in frequency
            if mod(Nch, 2) ~= 0
                freqs = -(Nch-1)/2 : (Nch-1)/2; 
            else
                freqs = (1:Nch) - (Nch+1)/2;
            end
            
            wdm = f0 + df * freqs;
            
            % convert back to wavelength if needed
            if rescale
                wdm = convert.freq2lambda(wdm);        
            end
        end
    end
end

