%
% normPropagationConst.m
%
% This method calculates the normalized propagation constants of the 
% requested modes or of all the modes that propagates up to a given 
% normalized frequency. The normalized propagation constant is defined as
% 
%      b = (beta^2 - k_cl^2) / (k_co^2 - k_cl^2)
% 
% Possible syntaxes:
% 
%  [b, cof] = obj.normPropagationConst(modes)
%  [b, cof] = obj.normPropagationConst(modes, v)
%  [b, cof, modes] = obj.normPropagationConst(v)
%  [b, cof, modes] = obj.normPropagationConst(types)
%  [b, cof, modes] = obj.normPropagationConst(types, v)
%
% The first two syntaxes return the normalized propagation constants for 
% the modes in the FiberModes object 'modes', evaluated on the normalized 
% frequencies given by 'v' or specified by 'obj' itself. 'cof' is 
% the vector with the corresponding normalized cutoff frequencies. 
%
% The third to fifth syntaxes do the same but for all the modes (or for 
% those specified by 'types') evaluated on the normalized frequencies given 
% by 'v' or by 'obj' itself. Returns also 'modes', the list of propagating 
% modes. The argument 'type' can be 'hybrid', 'LP', 'HE', 'EH', 'TE', 'TM' 
% or any combination of them in a comma-separated string.
%
% In all cases 'b' is a matrix with many columns as the number of modes and
% many rows as the elements of v. The elements of b are set to NaN whenever
% they correspond to modes that do not propagate at the considered
% frequency.
%

function [b, cof, modes]=normPropagationConst(obj, varargin)

% The function tries to use the information stored in obj.NPC; if these are
% not sufficient, forces a new calculation and updates data in obj.NPC. 

if isempty(varargin) 
	error('SIF:syntax', 'Syntax error.')
end

if isa(varargin{1}, 'FiberModes')
	% first/second syntax
	% we have to calculate for specific modes
	modes=varargin{1};
	if modes.nmodes==0
		error('SIF:syntax', 'There are no modes!')
	end
	
	if length(varargin)==2
		v=varargin{2};
	else
		v=obj.normFrequency;
	end
else
	% third/fifth syntax
	% we have to calculate all the modes in the given range
	modes=FiberModes;
	
	% default types
	typestr='hybrid lp';
	
	% default freqs.
	v=[];
	
	for n=1:length(varargin)
		if isnumeric(varargin{n})
			% we are given the frequencies!
			v=varargin{n};
		elseif ischar(varargin{n})
			% we are given the types
			typestr=varargin{n};
		else
			error('SIF:syntax', 'Syntax error.')
		end
	end
	
	% the types
	types=gettypes(typestr);
	
	% if v is still empty, let's populate it with obj
	if isempty(v)
		v=obj.normFrequency;
	end	
end

% check the quality of v
if any(v)<0 || any(isinf(v)) || any(isnan(v))
	error('SIF:syntax', 'Syntax error.')
end


if obj.verbose
	if modes.nmodes>0
		fprintf('Calculating the propagation constants of %d modes in the range [%.2f, %.2f].\n', ...
			modes.nmodes, min(v), max(v))
	else
		fprintf('Calculating the propagation constants in the range [%.2f, %.2f].\n', ...
			min(v), max(v))
	end
end



%__________________________________________________________________________
%
% definition of the functions that give the PC we set to zero 
% the order is the same as in FiberModes.labels

eqcar=cell(1,5);

% % auxiliary functions
% effe=@(b,v) (1./(v.^2.*(1-b.^2)) + 1./(v.^2 .* b.^2)) .* ...
%     (obj.nco^2./(v.^2.*(1-b.^2)) + obj.ncl^2./(v.^2 .* b.^2));
% R=@(n,x) (besselj(n-1,x) - besselj(n+1,x))./(2*x.*besselj(n,x));
% Q=@(n,x) -(besselk(n-1,x) + besselk(n+1,x))./(2*x.*besselk(n,x));
% 
% % HE modes (n>=1)
% eqcar{1}=@(b,v,n) obj.nco^2 * R(n, v.*sqrt(1-b.^2)) + ...
% 	(obj.nco^2 + obj.ncl^2) * Q(n, v.*b)/2 + ...
%     sqrt(obj.NA^4 * Q(n, v.*b).^2 / 4 + obj.nco^2 * n^2 * effe(b,v));
% 
% % EH modes (n>=1)
% eqcar{2}=@(b,v,n) obj.nco^2 * R(n, v.*sqrt(1-b.^2)) + ...
% 	(obj.nco^2 + obj.ncl^2) * Q(n, v.*b)/2 - ...
%     sqrt(obj.NA^4 * Q(n, v.*b).^2 / 4 + obj.nco^2 * n^2 * effe(b,v));
% 
% % TE modes
% eqcar{3}=@(b,v,n) besselj(2, v.*sqrt(1-b.^2))./besselj(0, v.*sqrt(1-b.^2)) + ...
%     besselk(2, v.*b)./besselk(0, v.*b);
% 
% % TM modes
% eqcar{4}=@(b,v,n) obj.nco^2*(besselj(2, v.*sqrt(1-b.^2))./besselj(0, v.*sqrt(1-b.^2)) + 1) + ...
%     obj.ncl^2*(besselk(2, v.*b)./besselk(0, v.*b) - 1);
% 
% % LP modes
% eqcar{5}=@(b,v,n) v.*sqrt(1-b.^2).*besselj(n+1, v.*sqrt(1-b.^2)) - ...
%         v.*b.*besselj(n, v.*sqrt(1-b.^2)).*besselk(n+1, v.*b)./besselk(n, v.*b);

% auxiliary functions
effe=@(b,v) (1./(v.^2.*(1-b)) + 1./(v.^2 .* b)) .* ...
    (obj.nco^2./(v.^2.*(1-b)) + obj.ncl^2./(v.^2 .* b));
R=@(n,x) (besselj(n-1,x) - besselj(n+1,x))./(2*x.*besselj(n,x));
Q=@(n,x) -(besselk(n-1,x) + besselk(n+1,x))./(2*x.*besselk(n,x));

% HE modes (n>=1)
eqcar{1}=@(b,v,n) obj.nco^2 * R(n, v.*sqrt(1-b)) + ...
	(obj.nco^2 + obj.ncl^2) * Q(n, v.*sqrt(b))/2 + ...
    sqrt(obj.NA^4 * Q(n, v.*sqrt(b)).^2 / 4 + obj.nco^2 * n^2 * effe(b,v));

% EH modes (n>=1)
eqcar{2}=@(b,v,n) obj.nco^2 * R(n, v.*sqrt(1-b)) + ...
	(obj.nco^2 + obj.ncl^2) * Q(n, v.*sqrt(b))/2 - ...
    sqrt(obj.NA^4 * Q(n, v.*sqrt(b)).^2 / 4 + obj.nco^2 * n^2 * effe(b,v));

% TE modes
eqcar{3}=@(b,v,n) besselj(2, v.*sqrt(1-b))./besselj(0, v.*sqrt(1-b)) + ...
    besselk(2, v.*sqrt(b))./besselk(0, v.*sqrt(b));

% TM modes
eqcar{4}=@(b,v,n) obj.nco^2*(besselj(2, v.*sqrt(1-b))./besselj(0, v.*sqrt(1-b)) + 1) + ...
    obj.ncl^2*(besselk(2, v.*sqrt(b))./besselk(0, v.*sqrt(b)) - 1);

% LP modes
eqcar{5}=@(b,v,n) v.*sqrt(1-b).*besselj(n+1, v.*sqrt(1-b)) - ...
        v.*sqrt(b).*besselj(n, v.*sqrt(1-b)).*besselk(n+1, v.*sqrt(b))./besselk(n, v.*sqrt(b));


%__________________________________________________________________________
%
% calculates the COF and determines the modes we have to consider

if modes.nmodes==0
	% we have to find also which mode can propagate in the specified 
	% frequency range
	[cof, modes]=obj.normCutoffFreq(max(v), typestr);
else
	cof=obj.normCutoffFreq(modes);
end


%__________________________________________________________________________
%
% calculates the PC

% number of frequencies
nv=length(v);

% allocates the space for the result
b=nan(modes.nmodes,nv);

% calcolates the propagation constants by following the list of modes, 
% grouping them for same type and azimuthal order

% sort the mode list by mode type
sm=FiberModes(modes);
indx=sm.sort;
scof=cof(indx);
% de-sorting indeces
[~, invindx]=sort(indx);
	

% cycle over modes with the same type and azimuth order
i1=1;
while i1<=sm.nmodes
	
	% determine the range of modes of the same type and azimuth order
	i2=i1-1;
	while i2<sm.nmodes && all(sm.modes(i2+1,1:2)==sm.modes(i1,1:2))
		i2=i2+1;
	end
	
	if obj.verbose
		m1=sm(i1);
		m2=sm(i2);
		fprintf('Analyzing modes from %s to %s...',  ...
			m1.string, m2.string)
	end
	
	% cycle over frequency of interest
	for k=1:nv
		
		% find the number of modes of the specific kind in analysis that
		% propagates at the current frequency
		h=find(scof(i1:i2)<=v(k), 1, 'last');
		if ~isempty(h)
			% number of modes, i.e. radial order of the highest mode
			p=sm.modes(i1+h-1,3);
		else
			% no modes are propagating
			p=0;
		end
		
		% search the zeros only if at least one of the mode is above cutoff
		if p>0
					
			keyboard
			betas=thezeros(@(x) eqcar{sm.modes(i1,1)}(x, v(k), sm.modes(i1,2)), p);
		
			% by the following sorting, the j-th element of betas will be the
			% PC of the mode with radial order j
			betas=sort(betas, 2, 'descend');
			
			% stores the right solutions in the right place
			for j=1:length(betas)
				% scan the modes looking for the one with the right radial
				% order
				for i=i1:i2
					if sm.modes(i,3)==j
						b(i,k)=betas(j);
						break
					end
				end
			end
		end
	end
	
	if obj.verbose
		fprintf(' done.\n')
	end
	
	% moves to the next case
	i1=i2+1;
	
end

% re-sort the COFs as in the input order
b=b(invindx,:);





%
%__________________________________________________________________________
%__________________________________________________________________________
%__________________________________________________________________________
%__________________________________________________________________________
%


function b0=thezeros(fun, p)
	
% Determine the zeros of function fun within the range [0 1]. If we are
% here, it's because there are at least p>0 progating modes.

% a parameter...
os=optimset('tolx', 1e-7);

% determines the zeros coarsely
np=1000;
% explicitely avoids b=1
b=(0:np-1)/np;

val=fun(b);
dsval=diff(sign(val));
I=find(dsval~=0 & ~isnan(dsval));

keyboard

if isempty(I)
	
	% the coarse approach has not found any zero, but we know there are at
	% least p of them. Actually this shoudl happen only for p==1
	
	% make a brutal search
	b0=fminbnd(@(x) log(abs(fun(x))), 0, 1, os);
	
else	
	
	b0=zeros(size(I));
	
	% refine the zeros by looking for the minimum in a bounded range
	% (functions like fzero goes outsiede the range [0 1] causing complex
	% numbers)
	ni=length(I);
	for i=1:ni
		
		% determine a good range
		if i==1
			bmin=0;
		else
			bmin=b(I(i-1));
		end
		if i==ni
			bmax=1;
		else
			bmax=b(I(i+1));
		end
		
		% finds the minimum
		b0s=fminbnd(@(x) log(abs(fun(x))), ...
			b(I(i)) - (b(I(i))-bmin)/2, ...
			b(I(i)) + (bmax-b(I(i)))/2, ...
			os);
		try
			b0(i)=b0s;
		catch
			keyboard
		end
	end
end


%__________________________________________________________________________
%



function types=gettypes(str)

% Determine the requested types

% cases and corresponding values
cases={'hybrid', 'he', 'eh', 'te', 'tm', 'lp'};
values={1:4, 1, 2, 3, 4, 5};

% main cycle
types=[];
for n=1:length(cases)
	if ~isempty(regexp(lower(str), cases{n}, 'match'))
		types=union(types, values{n});
	end
end

% coverts any TM in TE so that it will be performed
types(types==4)=3;

% and avoids repetitions
types=unique(types);

			