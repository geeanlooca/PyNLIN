function dP = raman_ode(z, P, params)
        %RAMAN_ODE Function describing the set of differential equations used for Raman
        % amplifiers.
        % First N values in vector P are the signal powers, while the last M
        % values are the pump powers

        dP = params.direction .* (-params.alpha + params.G * P) .* P;
end

