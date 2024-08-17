function dP = raman_ode(z, P, params)
        %RAMAN_ODE Function describing the set of differential equations used for Raman
        % amplifiers.
        % First N values in vector P are the signal powers, while the last M
        % values are the pump powers
        
        N = params.signals;
        M = params.pumps;
        freqs = params.freqs;
        gain = params.gain;
        dP = zeros(N+M, 1);
        
        losses = - params.alpha(:) .* P(:);
        
        % signal equations
        for i = 1:N+M
            
            % signal-signal interaction
            pumps = 0;
            for j = 1:N+M
                freq_factor = freqs(i) / freqs(j);
                pumps = pumps + freq_factor * gain(i, j) * P(i) * P(j);
            end
            
            dP(i) = params.direction(i) .* (losses(i) + pumps);
        end
end

