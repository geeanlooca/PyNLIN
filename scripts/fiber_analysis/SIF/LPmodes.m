classdef LPmodes < handle
    %LPMODES Computes the LP modes of a given step-index fiber
    %
    
    properties
        rco;        % Core radius
        nco;        % Core refractive index
        ncl;        % Cladding refractive index
        Delta;      % Relative index differente
        NA;         % Numerical Aperture
    end
    
    methods
        function obj = LPmodes(coreRadius, nCore, nCladding)
            %LPMODES Construct an instance of this class
            %
            %   coreRadius: the radius of the core
            %   nCore:  refractive index of the core
            %   nCladding: refractive index of the cladding
            
            p = inputParser;
            p.addRequired('coreRadius', @(x) validateattributes(x, ...
                {'numeric'}, { 'real', 'scalar', 'positive', 'finite' }));
            p.addRequired('nCore', @(x) validateattributes(x, ...
                {'numeric'}, {'real', 'scalar', 'finite', '>=', 1}));
            p.addRequired('nCladding', @(x) validateattributes(x, ...
                {'numeric'}, {'real', 'scalar', 'finite', '>=', 1}));
            
            parse(p, coreRadius, nCore, nCladding);
            
            if (p.Results.nCore <= p.Results.nCladding)
                error('nCore must be greater than nCladding');
            end
            
            obj.nco = p.Results.nCore;
            obj.ncl = p.Results.nCladding;
            obj.rco = p.Results.coreRadius;
            obj.NA = sqrt(obj.nco^2 - obj.ncl^2);
            obj.Delta = obj.NA^2 /(obj.nco^2);
        end
        
        function [beta, v] = getPropagationConstants(obj, wavelength, varargin)
            %GETPROPAGATIONCONSTANTS Solve the characteristic equation to
            %obtain the propagation constants of the LP modes of the
            %specified fiber at the given wavelength.
            %
            %   wavelength: the wavelength at which to compute the
            %   propagation constants. Can be a vector.
            
            p = inputParser;
            p.addRequired('wavelength', @(x) validateattributes(x, ...
                {'numeric'}, {'real', 'positive', 'finite', 'vector'}));
            
            p.addParameter('normalize', true, @(x) validateattributes(x, ...
                {'logical'}, {'scalar'}));
            
            parse(p, wavelength, varargin{:});
            
            % solve the equation
            J = @(n, x) besselj(n, x);
            Y = @(n, x) bessely(n, x);
            K = @(n, x) besselk(n, x);
            I = @(n, x) besseli(n, x);
            
            v = obj.getNormalizedFrequency(wavelength);
            
            kco = 2 * pi ./ p.Results.wavelength * obj.nco;
            kcl = 2 * pi ./ p.Results.wavelength * obj.ncl;
            
            b0 = 0;
            b1 = 1;
            db = 0.05;
            
%             opt = optimset('Display', 'iter');
            beta = zeros(length(wavelength), 1);
            n = 0;
                        
            for i = 1 : length(wavelength)
                beta(i) = fminbnd( @(x) charEquation(x, v(i), n), b0, b1);                
                b0 = beta(i) - db/2;
                b1 = beta(i) + db/2;
                
            end
            
            function res = charEquation(b, v, n)
                v1b = v * sqrt(1-b);
                vb = v * sqrt(b);
                lhs = v1b * J(n+1, v1b) / J(n, v1b);
                rhs = vb * K(n+1, vb) / K(n, vb);
                res = lhs - rhs;
            end
            
        end
        
        function v = getNormalizedFrequency(obj, wavelength)
            p = inputParser;
            p.addRequired('wavelength', @(x) validateattributes(x, ...
                {'numeric'}, {'real', 'positive', 'finite', 'vector'}));
            
            parse(p, wavelength);
            
            v = 2*pi*obj.rco./p.Results.wavelength * obj.NA;
            
        end
    end
end

