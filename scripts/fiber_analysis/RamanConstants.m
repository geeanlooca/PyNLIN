classdef RamanConstants
    %RAMANCONSTANTS Collection of physical constants useful for Raman
    %amplification.
    %   This class contains the definition of some useful constants or
    %   typical values for some parameters used in the context of fiber
    %   Raman amplifiers.
    
    properties (Constant)
        typicalGain = 7e-14;    % Typical value for Raman gain [m/W]
        chi3Si = 0.732;         % Third-order susceptibility of Silica, taken from https://doi.org/10.1364/AO.21.003221
    end
end
