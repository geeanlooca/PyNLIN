function mask = LPmodeProfile(modes)
%LPMODEPROFILE Computes the field distribution for the LP modes given as
%input
    
% build the grid

end

function E = getRadialMask(l, r, chico, chicl, a)
    core_idx = abs(r) <= a;
    rcore = r(core_idx);
    clad_idx = abs(r) > a;
    rclad = r(clad_idx);
    E = zeros(size(r));
    Ecore = besselj(l, chico * rcore) ./ besselj(l, chico * a);
    Eclad = besselk(l, chicl * rclad) ./ besselk(l, chicl * a);
    E(core_idx) = Ecore;
    E(clad_idx) = Eclad;
end