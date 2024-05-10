function f = OverlapIntegral(mode1,mode2, x, y)
%OVERLAPINTEGRAL Computes the overlap integrals between the two modes
%specified as input. 
%   
%   f = OVERLAPINTEGRAL(mode1, mode2) computes the overlap integ
%

    I1 = abs(mode1).^2;
    I2 = abs(mode2).^2;
    num = trapz(y, trapz(x, I1 .* I2));
    den1 = trapz(y, trapz(x, I1));
    den2 = trapz(y, trapz(x, I2));
    f = num ./ ( den1 .* den2);
end