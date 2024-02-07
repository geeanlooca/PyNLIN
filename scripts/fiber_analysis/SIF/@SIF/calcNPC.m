%
% caclNPC
%
% Calculate the normalized propagation constant of a mode from it's cutoff
% up to the specified frequency.
%
%
%-----------------------    W A R N I N G    ------------------------------
%
% Due to considerations about numerical stability, the NPC is defined as
%
%      b = sqrt( (beta^2 - k_cl^2) / (k_co^2 - k_cl^2) )
%
% Note that this is different from what the function 'normPropagationConst' 
% returns!!!
%
%-----------------------    W A R N I N G    ------------------------------
%
%

function calcNPC(obj, mode, cof, vmax)


%__________________________________________________________________________
%
%  some general parameter

if mode.modes(1,1)==5 || mode.modes(1,1)==3 
	% the mode is an LP of TE; let use a fine step in v
	dv=0.001;
else
	% the mode is something else; the step is less fine
	dv=0.01;
end



%__________________________________________________________________________
%
% definition of the functions that give the PC we set to zero 
% the order is the same as in FiberModes.labels

eqcar=cell(1,5);

% auxiliary functions
effe=@(b,v) (1./(v.^2.*(1-b.^2)) + 1./(v.^2 .* b.^2)) .* ...
    (obj.nco^2./(v.^2.*(1-b.^2)) + obj.ncl^2./(v.^2 .* b.^2));
R=@(n,x) (besselj(n-1,x) - besselj(n+1,x))./(2*x.*besselj(n,x));
Q=@(n,x) -(besselk(n-1,x) + besselk(n+1,x))./(2*x.*besselk(n,x));

% HE modes (n>=1)
eqcar{1}=@(b,v,n) obj.nco^2 * R(n, v.*sqrt(1-b.^2)) + ...
	(obj.nco^2 + obj.ncl^2) * Q(n, v.*b)/2 + ...
    sqrt(obj.NA^4 * Q(n, v.*b).^2 / 4 + obj.nco^2 * n^2 * effe(b,v));

% EH modes (n>=1)
eqcar{2}=@(b,v,n) obj.nco^2 * R(n, v.*sqrt(1-b.^2)) + ...
	(obj.nco^2 + obj.ncl^2) * Q(n, v.*b)/2 - ...
    sqrt(obj.NA^4 * Q(n, v.*b).^2 / 4 + obj.nco^2 * n^2 * effe(b,v));

% TE modes
eqcar{3}=@(b,v,n) besselj(2, v.*sqrt(1-b.^2))./besselj(0, v.*sqrt(1-b.^2)) + ...
    besselk(2, v.*b)./besselk(0, v.*b);

% TM modes
eqcar{4}=@(b,v,n) obj.nco^2*(besselj(2, v.*sqrt(1-b.^2))./besselj(0, v.*sqrt(1-b.^2)) + 1) + ...
    obj.ncl^2*(besselk(2, v.*b)./besselk(0, v.*b) - 1);

% LP modes
eqcar{5}=@(b,v,n) v.*sqrt(1-b.^2).*besselj(n+1, v.*sqrt(1-b.^2)) - ...
        v.*b.*besselj(n, v.*sqrt(1-b.^2)).*besselk(n+1, v.*b)./besselk(n, v.*b);

% % auxiliary functions
% effe=@(b,v) (1./(v.^2.*(1-b)) + 1./(v.^2 .* b)) .* ...
%     (obj.nco^2./(v.^2.*(1-b)) + obj.ncl^2./(v.^2 .* b));
% R=@(n,x) (besselj(n-1,x) - besselj(n+1,x))./(2*x.*besselj(n,x));
% Q=@(n,x) -(besselk(n-1,x) + besselk(n+1,x))./(2*x.*besselk(n,x));
% 
% % HE modes (n>=1)
% eqcar{1}=@(b,v,n) obj.nco^2 * R(n, v.*sqrt(1-b)) + ...
% 	(obj.nco^2 + obj.ncl^2) * Q(n, v.*sqrt(b))/2 + ...
%     sqrt(obj.NA^4 * Q(n, v.*sqrt(b)).^2 / 4 + obj.nco^2 * n^2 * effe(sqrt(b),v));
% 
% % EH modes (n>=1)
% eqcar{2}=@(b,v,n) obj.nco^2 * R(n, v.*sqrt(1-b)) + ...
% 	(obj.nco^2 + obj.ncl^2) * Q(n, v.*sqrt(b))/2 - ...
%     sqrt(obj.NA^4 * Q(n, v.*sqrt(b)).^2 / 4 + obj.nco^2 * n^2 * effe(sqrt(b),v));
% 
% % TE modes
% eqcar{3}=@(b,v,n) besselj(2, v.*sqrt(1-b))./besselj(0, v.*sqrt(1-b)) + ...
%     besselk(2, v.*sqrt(b))./besselk(0, v.*sqrt(b));
% 
% % TM modes
% eqcar{4}=@(b,v,n) obj.nco^2*(besselj(2, v.*sqrt(1-b))./besselj(0, v.*sqrt(1-b)) + 1) + ...
%     obj.ncl^2*(besselk(2, v.*sqrt(b))./besselk(0, v.*sqrt(b)) - 1);
% 
% % LP modes
% eqcar{5}=@(b,v,n) v.*sqrt(1-b).*besselj(n+1, v.*sqrt(1-b)) - ...
%         v.*sqrt(b).*besselj(n, v.*sqrt(1-b)).*besselk(n+1, v.*sqrt(b))./besselk(n, v.*sqrt(b));


	
%__________________________________________________________________________
%
% check if the mode is already in the data base

m=obj.NPC.modes.find(mode);

if isempty(m)
	
	% the mode is not in the data base, let's add it
	obj.NPC.modes.add(mode);
	
	% read the new index
	m=obj.NPC.modes.find(mode);
	
	% add the cutoff frequency
	obj.NPC.cof(m)=cof;
	
	% add v (include cof and vmax)
	obj.NPC.v{m}=union(cof:dv:vmax, vmax);
	
	% creates the space for b
	obj.NPC.b{m}=zeros(size(obj.NPC.v{m}));
	
	% index to the starting frequency (for the analysis)
	i0=1;
	
else
	
	% the mode is there
	
	% let's see how many frequencies we must add
	nv=obj.NPC.v{m};
	% index to the starting frequency (for the analysis)
	i0=length(nv);
	av=union(nv(end)+dv:dv:vmax, vmax);
	obj.NPC.v{m}=[nv av];
	
	% allocate the space for b
	nb=obj.NPC.b{m};
	nb=[nb zeros(size(av))];
	obj.NPC.b{m}=nb;
	
end


	
%__________________________________________________________________________
% 
% executes calculation starting from after the cof
% calcola le costanti normalizzate b
os=optimset('tolx', 1e-7);
db=.05;
   
% current mode
cm=obj.NPC.modes(m);
% mode type
k=cm.modes(1,1);
% azimuthal index
n=cm.modes(1,2);

% look for the right b in an interval wide db around the value of b found 
% at the previous iteration
for i=i0+1:length(obj.NPC.v{m})
	b0=max(0, obj.NPC.b{m}(i-1)-db/2);
	b1=min(1, obj.NPC.b{m}(i-1)+db/2);
	obj.NPC.b{m}(i)=fminbnd(@(x) log(abs(eqcar{k}(x, obj.NPC.v{m}(i), n))), b0, b1, os);
end
    




