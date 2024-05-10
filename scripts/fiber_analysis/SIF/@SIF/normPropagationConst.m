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

% try to convert the first argument in a fiber mode
if ischar(varargin{1})
	try
		% converting input in Fibermodes
		varargin{1}=FiberModes(varargin{1});
	catch err
		if strcmp(err.identifier, 'FiberModes:wrongIndex')
			% rethrow the error
			rethrow(err)
		end
	end
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
% calculates the NPC

% number of frequencies
nv=length(v);

% allocates the space for the result
b=nan(modes.nmodes,nv);

% calculates the propagation constants by following the list of modes.
% cycle over modes with the same type and azimuth order
for m=1:modes.nmodes
	
	% the current mode
	cm=modes(m);
	
	if obj.verbose
		fprintf('Analyzing mode %s.\n', cm.string);
	end
	
	% find the indeces of the normalized frequencies at which the current 
	% mode propagates 
	iv=find(v>=cof(m));
	
	% check if the mode can actually propagate at the sought frequencies
	if ~isempty(iv)		
		% yet it can!

		% first check if the mode is in NPC
		im=obj.NPC.modes.find(cm);
		
		if isempty(im)
			
			% the mode is not there			
			if obj.verbose
				fprintf('The mode %s is not in the NPC database\n', cm.string);
			end
		
			% let's force its calculation up-to the highest frequency
			obj.calcNPC(cm, cof(m), max(v));
			
			% redetermine im
			im=obj.NPC.modes.find(cm);
			
		else
			
			% yes it's in there
			if obj.verbose
				fprintf('The mode %s is in the NPC database...', cm.string);
			end
			
			% let's see if the requested frequencies fall outside the 
			% available ones
			jv=find(v>max(obj.NPC.v{im}));
			
			if ~isempty(jv)
				
				% we need to calculate some of the frequency that are
				% outside the known range
				if obj.verbose
					fprintf(' but some frequencies are missing.\n');
				end
				
				% let's force its calculation up-to the highest frequency
				obj.calcNPC(cm, cof(m), max(v));
			
			else
				
				% we already have all the information we need
				if obj.verbose
					fprintf(' and all the frequencies are there.\n');
				end
				
			end
			
		end
		
		% at this stage the data in NPC can be safely used to calculated
		% the requested propagation constant by interpolation 
		
		% ----- W A R N I N G ---->  remember that b = NPC^2 !!! 
		b(m,iv)=interp1(obj.NPC.v{im}, obj.NPC.b{im}, v(iv), 'spline').^2;
	end
end
				
			
			



%
%__________________________________________________________________________
%__________________________________________________________________________
%__________________________________________________________________________
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

			