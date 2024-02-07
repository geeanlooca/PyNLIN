%
% normCutoffFreq.m
%
% This method calculates the normalized cutoff frequecies of the requested
% modes or of all the modes that propagates up to a given normalized
% frequency.
%
%  cof = obj.normCutoffFreq(modes)
%  [cof, modes] = obj.normCutoffFreq(vmax, types)
%  [cof, modes] = obj.normCutoffFreq(types)
%
% The first syntax returns the normalized cutoff frequencies for the modes
% in the FiberModes object 'modes'; 'cof' is a vector with length equal to
% the number of modes.
% The second syntax calculates the cutoff frequencies of all the modes up
% to the normalized frequency 'vmax'; cutoff frequencies and corresponding
% modes are returned in 'cof' and 'modes', respectivelly, sorted by
% increasing cutoff frequency.
% The third syntax calculate the cutoff frequency of all the modes that can
% propagate in the frequency range specified with object 'obj'.
% In the last two syntaxes the string 'types' limits the analysis to only
% some types of mode. It can be 'hybrid', 'LP', 'HE', 'EH', 'TE', 'TM' or
% any combination of them in a comma-separated string.
%

function [cof, modes]=normCutoffFreq(obj, varargin)


if isempty(varargin) || ischar(varargin{1})
	% third syntax
	% we have to calculate the COF of the modes that can propagate in the
	% specified frequency range.
	modes=FiberModes;
	vmax=max(obj.normFrequency);
	% determine the types
	if isempty(varargin)
		% considers all the types
		types=1:5;
	else
		types=gettypes(varargin{1});
	end
else
	if isa(varargin{1}, 'FiberModes')
		% first syntax
		% we have to calculate the cof for specific modes
		modes=varargin{1};
		vmax=Inf;
	else
		% second syntax
		% we have to calculate all the COF up to vmax
		modes=FiberModes;
		vmax=varargin{1};
		if ~isscalar(vmax) || vmax<0 || isinf(vmax) || isnan(vmax)
			error('SIF:syntax', 'Syntax error.')
		end
		% determine the types
		if length(varargin)==1
			% considers all the types
			types=1:5;
		else
			types=gettypes(varargin{2});
		end
	end
end

if obj.verbose
	if isinf(vmax)
		fprintf('Calculating the cutoff frequencies of %d modes.\n', ...
			modes.nmodes)
	else
		fprintf('Calculating the cutoff frequencies for v <= %.2f.\n', vmax)
	end
end



%__________________________________________________________________________
%
% definition of the functions that give the COF we set to zero
% the order is the same as in FiberModes.labels

fcut=cell(1,5);
fcut{1}=@(x,n) x.*besselj(n,x) - ...
	(n-1)*besselj(n-1,x)*(1+obj.nco^2/obj.ncl^2);       % HE  (n>=1)
fcut{2}=@(x,n) besselj(n,x);                            % EH  (n>=1)
fcut{3}=@(x,n) besselj(0,x);                            % TE
fcut{4}=@(x,n) besselj(0,x);                            % TM
fcut{5}=@(x,n) x.*besselj(n+1,x) - 2*n*besselj(n,x);    % LP



%__________________________________________________________________________
%
% calculates the COF

if isinf(vmax)


	%----------------------------------------------------------------------
	% we have a list of modes, let's follow it
	%----------------------------------------------------------------------

	% sort the mode list by mode type
	sm=FiberModes(modes);
	indx=sm.sort;
	[~, invindx]=sort(indx);

	% allocates memory for the COFs
	cof=zeros(1,sm.nmodes);

	% cycle over modes with of the same type and azimuth order
	i1=1;
	while i1<=sm.nmodes

		% determine the range of modes of the same type and azimuth order
		i2=i1-1;
		while i2<sm.nmodes && all(sm.modes(i2+1,1:2)==sm.modes(i1,1:2))
			i2=i2+1;
		end

		if obj.verbose
			fprintf('Analyzing modes from %d to %d...', i1, i2)
		end


		% estimate a range of normalized frequency where to look for the
		% COF; if not enough solution are found on the range, repeat the
		% analysis on a second range

		% starting estimate
		Dv=1.3*(sm.modes(i1,2) + sm.modes(i2,3));
		vmin=0;
		vmax=Dv;

		maxp=sm.modes(i2,3);   % maximum radial order
		v0s=[];
		while 1
			% search the zeros...
			v0=thezeros(@(x) fcut{sm.modes(i1,1)}(x, sm.modes(i1,2)), ...
				vmin, vmax, sm(i1));
			v0s=[v0s v0];

			% if needed, increases the range
			if length(v0s)<maxp
				% extends the range
				vmin=vmax;
				vmax=vmax+Dv;
				if ~isempty(v0) && vmin==v0(end)
					% the zero is just at the border, let's avoid finding
					% it twice
					vmin=vmin+Dv/20;
				end

				if obj.verbose
					fprintf('|...')
				end
			else
				% we have done
				break
			end
		end

		if obj.verbose
			fprintf(' done.\n')
		end

		% stores the found COFs
		for i=i1:i2
			cof(i)=v0s(sm.modes(i,3));
		end

		% moves to the next case
		i1=i2+1;

	end

	% re-sort the COFs as in the input order
	cof=cof(invindx);


else


	%----------------------------------------------------------------------
	% we have a range to explore
	%----------------------------------------------------------------------

	% allocates the space for modes and COFs
	modes=FiberModes;
	cof=[];

	% cycle over mode types
	for m=types(1):types(end)

		if obj.verbose
			fprintf('Analyzing %s modes...\n', FiberModes.labels{m})
		end

		% TE and TM modes are degenerate at the cutoff, so the TM case is
		% jumped; its case is implicitely handled during the TE case
		if m~=4

			% lower azimuthal order from which to start
			if m==1 || m==2
				% HE and EH
				n=1;
			else
				% TE, TM, LP
				n=0;
			end

			% cycle over azimuthal order
			next=true;
			while next

				if obj.verbose
					fprintf('... with order %d ', n)
				end

				% finds the zeros...
				v0=thezeros(@(x) fcut{m}(x, n), 0, vmax, ...
					FiberModes(FiberModes.labels{m}, [n, 1]));

				if obj.verbose
					fprintf('found %d modes\n', length(v0))
				end

				% ...and  stores them
				if ~isempty(v0)
					cof=[cof, v0];
					modestr='';
					for k=1:length(v0)
						modestr=sprintf('%s %s(%d,%d)', modestr, ...
							FiberModes.labels{m}, n, k);
					end
					modes.add(modestr);

					% if we are doing TE modes, add the same results for
					% the TM
					if m==3
						cof=[cof, v0];
						modestr='';
						for k=1:length(v0)
							modestr=sprintf('%s %s(%d,%d)', modestr, ...
								FiberModes.labels{4}, n, k);
						end
						modes.add(modestr);
					end

					% determine the next move
					if m==3
						% TE and TM modes have only azimuth uqual to 0
						next=false;
					else
						% consider the higher azimuthal order
						n=n+1;
					end

				else

					% we have to move to the next mode type
					next=false;

				end
			end
		end
	end

	% sorts the mode by increasing COF
	[cof, i]=sort(cof);
	modes=modes(i);

end



%
%__________________________________________________________________________
%__________________________________________________________________________
%__________________________________________________________________________
%__________________________________________________________________________
%


function v0=thezeros(fun, vmin, vmax, mode)

% Determine the zeros of function fun within the range [vmin vmax];
% discards the solution at zero depending on the considered mode.

% determines the zeros coarsly
v=vmin+(0:200)*(vmax-vmin)/200;
val=fun(v);
dsval=diff(sign(val));
I=find(dsval~=0);

% refine the zeros
v0=zeros(size(I));
for i=1:length(I)
	v0(i)=fzero(fun, v(I(i)));
end
v0=unique(v0);

% the solution v0==0 is valid only for HE with n==1 and LP with n==0
if ~isempty(v0) && v0(1)==0 && ...
		~(mode.modes(1)==1 &&  mode.modes(2)==1) && ...
		~(mode.modes(1)==5 &&  mode.modes(2)==0)
	% discards the "zero" zero
	v0=v0(2:end);
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
