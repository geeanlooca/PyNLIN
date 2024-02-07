classdef StepIndexFiber < handle
    %STEPINDEXFIBER Wrapper class for the definition of a step-index
    %optical fiber.
    %
    
    properties
        ncl;        % Refractive index of the cladding
        nco;        % Refractive index of the core
        rco;        % Radius of the core [m]
        V;          % Normalized frequency
        wavelength; % The wavelength [m]
        frequency;  % The frequency [Hz]
        NA;         % Numerical aperture
        modeType = 'lp'; % Type of fiber modes to consider
        exp2  = 10; % power-of-2 exponent determining the density of the integration grid
        length;     % Length of the fiber in meters
        RamanGain; % Raman gain coefficient
    end
    
    properties (GetAccess=public, SetAccess=private)
        sif = [];   % Step-Index Fiber object;
        COF;        % Cutoff frequencies;
        normCOF;    % Normalized cutoff frequencies;
        modes = []  % The modes
        normPropConst;  % Normalized propagation constants;
        OI;         % Overlap Integrals
        x; y;       % Spatial grid over which the mode profile is defined
        modeProfiles  {};
        Aeff;
    end
    
    methods
        function obj = StepIndexFiber(varargin)
            %StepIndexFiber Construct an instance of this class
            %   Defines the fiber geometry:
            %
            %   obj = StepIndexFiber(ncore, ncladding, rcore), where
            %       ncore (ncladding) is the refractive index of the core
            %       (cladding) and rcore is the core radius in meters.
            
            parser = inputParser;
            parser.addRequired('ncore', @(x) validateattributes(x, ...
                {'numeric'}, {'scalar', 'real', '>=', 1}));
            
            parser.addRequired('ncladding', @(x) validateattributes(x, ...
                {'numeric'}, {'scalar', 'real', '>=', 1}));
            
            parser.addRequired('rcore', @(x) validateattributes(x, ...
                {'numeric'}, {'scalar', 'real', 'positive'}));
            
            parse(parser, varargin{:});
            
            if parser.Results.ncore <= parser.Results.ncladding
                error(['The core refractive index must be strictly' ...
                    ' smaller than the cladding refractive index.']);
            end
            
            % Force a sensible default value
            obj.RamanGain = RamanConstants.typicalGain;            
            
            
            obj.nco = parser.Results.ncore;
            obj.ncl = parser.Results.ncladding;
            obj.rco = parser.Results.rcore;
            obj.NA = sqrt(obj.nco^2 - obj.ncl^2);
            obj.exp2 = 10;
        end
        
        function set.exp2(obj, value)
            pts = 2^value;
            k = 3;
            obj.x = linspace(-obj.rco * k, obj.rco * k, pts);
            obj.y = linspace(-obj.rco * k, obj.rco * k, pts);
        end
        
        function set.wavelength(obj, value)
            obj.wavelength = value;
            obj.frequency = constants.c0 ./ obj.wavelength;
            obj.V = 2 * pi * obj.rco ./ obj.wavelength * obj.NA;
            obj.modeProfiles = [];
            
            % Update the SIF object
            obj.sif = SIF('nco', obj.nco, 'ncl', obj.ncl, ...
                'radius', obj.rco, 'wavelength', obj.wavelength);
            
            % Recompute modes
            [obj.COF, obj.modes] = obj.sif.cutoffFreq(obj.modeType);            
            obj.normCOF = obj.sif.normCutoffFreq();
            obj.normPropConst = obj.sif.normPropagationConst(obj.modes)';
        end
        
        function value = get.NA(obj)
            value = obj.NA;
        end
        
        function value = get.wavelength(obj)
            value = obj.wavelength;
        end
        
        function value = get.frequency(obj)
            value = obj.frequency;
        end
        
        function value = get.V(obj)
            value = obj.V;
        end
        
        % Number of modes without degeneracies
        function value = modeCount(obj)
            if ~isempty(obj.sif)
                value = obj.modes.nmodes;
            else
                error('Wavelength not set.');
            end
        end
        
        % Number of modes counting degeneracies
        function value = modeDegCount(obj)
            if ~isempty(obj.sif)
                value = obj.modes.ndegmodes;
            else
                error('Wavelength not set.');
            end
        end        
        
        % Obtain the 2D mode profile for the selected mode.
        % i is the index of the mode in the obj.modes structure
        function computeModeProfiles(obj)
            
            if ~isempty(obj.modes)
                % Store the profiles
                obj.modeProfiles = cell(obj.modes.nmodes, 2);                
                
                x = obj.x;
                y = obj.y;                
                
                for i = 1:obj.modes.nmodes                    
                    m = obj.modes(i);
                    ndeg = m.ndegmodes;
                    type = m.labels{m.modes(1)};
                    
                    l = m.modes(2);
                    p = m.modes(3);
                    
%                     fprintf('LP%d%d. Degenerations: %d\n', l, p, m.ndegmodes);
                    
                    propconst = obj.normPropConst(i);
                    
                    chicore = obj.V * sqrt(1-propconst) / obj.rco;
                    chiclad = obj.V * sqrt(propconst) / obj.rco;
                    
                    [X, Y] = ndgrid(x, y);
                    [THETA, R] = cart2pol(X, Y);
                    phasemaskCOS = cos(l * THETA);
                    phasemaskSIN = sin(l * THETA);                    
                    radialmask = obj.getRadialMask(l, R, chicore, chiclad, obj.rco);
                    
                    obj.modeProfiles{i, 1} = radialmask .* phasemaskCOS;                    
                    obj.modeProfiles{i, 2} = radialmask .* phasemaskSIN;
                end
            else
                error('Wavelength not set');
            end
        end
        
        function value = getModeProfiles(obj)
            if isempty(obj.modeProfiles)
                obj.computeModeProfiles();
            end
            
            value = obj.modeProfiles;
        end
        
        function computeOverlapIntegrals(obj)            
            obj.OI = zeros(obj.modes.nmodes);
            
            if isempty(obj.modeProfiles)
                obj.computeModeProfiles;
            end
            
            for i = 1 : obj.modes.nmodes
                for j = 1 : obj.modes.nmodes                
                    obj.OI(i, j) = obj.OverlapIntegral(obj.modeProfiles{i, 1}, ...
                        obj.modeProfiles{j, 1}, obj.x, obj.y);
                end
            end
        end
        
        function value = getOverlapIntegrals(obj)
            if isempty(obj.OI)
                obj.computeOverlapIntegrals();
            end
            
            value = obj.OI;
        end
        
        function value = effectiveArea(obj)
            if isempty(obj.OI)
                obj.computeOverlapIntegrals;
            end
            
            % Inverse of the overlap integral for the LP01 mode
            value = 1/obj.OI(1, 1);
        end
        
    end
    
    methods (Access=private)
        function E = getRadialMask(obj, l, r, chico, chicl, a)
            core_idx = abs(r) <= a;
            rcore = r(core_idx);
            clad_idx = abs(r) > a;
            rclad = r(clad_idx);
            E = zeros(size(r));
            Ecore = besselj(l, chico .* rcore) ./ besselj(l, chico .* a);
            Eclad = besselk(l, chicl .* rclad) ./ besselk(l, chicl .* a);
            E(core_idx) = Ecore;
            E(clad_idx) = Eclad;
        end
        
        function f = OverlapIntegral(obj, mode1,mode2, x, y)
            %OVERLAPINTEGRAL Computes the overlap integrals between the two modes
            %specified as input.
            %
            %   f = OVERLAPINTEGRAL(mode1, mode2) computes the overlap integ            
            
            I1 = abs(mode1).^2;
            I2 = abs(mode2).^2;
            num = trapz(y, trapz(x, I1 .* I2));
            den1 = trapz(y, trapz(x, I1));
            den2 = trapz(y, trapz(x, I2));
            f = num ./ (den1 .* den2);
        end
    end
end

