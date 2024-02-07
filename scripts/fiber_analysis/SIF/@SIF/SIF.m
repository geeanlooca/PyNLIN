%
% SIF - Step Index Fiber
%
% This class implements methods and properties to handle modes of step
% index fibers, including the calculation of their coupling coefficients.
%

classdef SIF < handle
	
	properties
		
		% fiber core radius [m]
		radius=[];
		
		% fiber cladding radius [m]
		radiusCL=[];
		
		% core refractive index
		nco=[];
		
		% cladding refractive index
		ncl=[];
		
		% wavelengths of interest (a vector) [m]
		wavelength=[];
		
		% verbosity (0, 1, 2...)
		verbose=0;
		
		% set to true to save the NPC database everytime and oject is
		% deleted
		updateNPC=false;
		
		%------------------------------------------------------------------
		% perturbation parameters
		%------------------------------------------------------------------
		
		% birefringence must be n_x - n_y
		birefringence=[];
		
		% 'deltaRatio' is the ratio between maximum core radius variation
		% and the core radius
		ellipticity=struct('deltaRatio', [], 'components', 'all');
		
		% 'radius' is the bending ratdius [m]
		bending=struct('radius', [], 'components', 'all');
		
		% 'rate' is the twist rate
		twist=struct('rate', [], 'g', 0.147);
		
	end
	

	properties (Constant)
		
		% speed of light in vacuum [m/s]
		c0=299792458;
	
		% vacuum permeability [H/m]
		mu0=4*pi*1e-7;
		
		% vacuum permittivity [F/m]
		epsilon0=1/(299792458^2 * 4*pi*1e-7);
		
	end
	
	
	properties (Dependent = true)
		
		% numerical aperture 
		NA=[];
			
		% frequencies of interest (a vector) [Hz]
		frequency=[];

		% normalized frequency of interest (a vector)
		% v = 2*pi*radius*NA/lambda 
		normFrequency=[];

	end
	
	
	properties (SetAccess = private)
		
		% The following property stores the calculated propagation
		% constants for each considered mode. The structure is "zeroed"
		% every time the refractive indeces are changed (expect for the LP
		% and TE modes whose normalized propagation constants do not depend
		% on those indeces).
		% This structure is needed because the zeros of the characteristic
		% equations can be safely found only following the dispersion curve
		% from the cutoff frequency, and it would be quite time consuming
		% to repeat that calculation every single time.
		NPC=struct('modes', FiberModes, 'cof', [], 'b', {{}}, 'v', {{}});
		
	end
	
	
	properties (Access = private)
		
		% name of the NPC file
		fileNPC='NPC.mat';
		
		% directory where the NPC is located
		dirNPC='';
		
		% this is the list of known perturbation! The elements must
		% correspond to the name of the corresponding property.
		perturbations={'ellipticity', 'bending', 'birefringence', 'twist'};
		
	end

	
	%
	%______________________________________________________________________
	%______________________________________________________________________
	%
	%
	

	methods
	
		%------------------------------------------------------------------
		% Costructor
		%------------------------------------------------------------------
		function obj=SIF(varargin)
			
			%
			% obj=SIF('prop', value, ...)
			%
			% The first syntax initialize the object with the specified
			% property values; 'prop' are the properties of the object.
			%
			% Note that object can be saved and reloaded with standard load
			% command.
			% 
			
			nai=length(varargin);
			if rem(nai,2) ~= 0 && nai~=1
				error('Iterator:syntax', 'Syntax error.')
			end

			if nai==1
				
				% we have a file name from which to load class parameters
				if ischar(varargin{1})
					obj.load(varargin{1});
				else
					error('SIF:syntax', 'The file name must be a string!')
				end
				
			else
				
				for n=1:2:nai
					
					prop=varargin{n};
					val=varargin{n+1};
					
					if ~ischar(prop)
						error('SIF:syntax', 'Property must be a string.')
					end
					
					% read the input parameters
					switch prop
						
						case {'radius', 'radiusCL', 'nco', 'ncl', 'wavelength', 'verbose', 'updateNPC'}
							obj.(prop)=val;
							
						case 'frequency'
							% converts from Hz to m
							obj.wavelength=SIF.c0./val;
							
						otherwise
							% error
							error('SIF:syntax', 'Property ''%s'' is unknown.', prop)
							
					end
				end
			end
			
			% determines the directory for the NPC
			mf=mfilename('fullpath');
			obj.dirNPC=fileparts(mf);
			
			% load the NPC
			obj.loadNPC;
			
		end
		
			
		%------------------------------------------------------------------
		% destructor
		%------------------------------------------------------------------
		function delete(obj)
			% save the current NPC database			
			
			if obj.updateNPC
				% a check
				if isempty(obj.NPC.cof)
					% probably this is a bug of matlab
					warning('SIF:bug', ...
						'Probably you are destroying an SIF object by assigning a new SIF object\n%s\n%s\n', ...
						'to the same varaible. For some unknown reason, in this case the destructor', ...
						'does not work properly and the NPC database will not be saved.')
				else
					obj.saveNPC;
				end
			end
		end
		
			
		%------------------------------------------------------------------
		% set.radius
		%------------------------------------------------------------------
		function set.radius(obj, val)
			% controls and set the radius
			
			if ~isscalar(val) || val<=0
				error('SIF:wrongvalue', 'Property ''radius'' must be a positive scalar.')
            end
            if ~isempty(obj.radiusCL) && val>obj.radiusCL
                error('SIF:wrongvalue', 'Property ''radius'' must smaller than ''radiusCL''.')
            end
			obj.radius=val;
		end
		
			
		%------------------------------------------------------------------
		% set.radiusCL
		%------------------------------------------------------------------
		function set.radiusCL(obj, val)
			% controls and set the cladding radius
			
			if ~isscalar(val) || val<=0
				error('SIF:wrongvalue', 'Property ''radiusCL'' must be a positive scalar.')
			end
			if ~isempty(obj.radius) && val<obj.radius
                error('SIF:wrongvalue', 'Property ''radiusCL'' must larger than ''radius''.')
            end
            obj.radiusCL=val;
		end
		
			
		%------------------------------------------------------------------
		% set.nco
		%------------------------------------------------------------------
		function set.nco(obj, val)
			% controls and set the core refractive index
			
			if ~isscalar(val) || val<1
				error('SIF:wrongvalue', 'Property ''nco'' must be a scalar larger than 1.')
			end
			if ~isempty(obj.ncl) && obj.ncl>=val
				error('SIF:wrongvalue', 'Property ''nco'' must be larger than ''ncl''.')
			end
			obj.nco=val;
			
			% reset the NPC structure
			obj.resetNPC;
		end
		
			
		%------------------------------------------------------------------
		% set.ncl
		%------------------------------------------------------------------
		function set.ncl(obj, val)
			% controls and set the cladding refractive index
			
			if ~isscalar(val) || val<1
				error('SIF:wrongvalue', 'Property ''ncl'' must be a scalar larger than 1.')
			end
			if ~isempty(obj.nco) && obj.nco<=val
				error('SIF:wrongvalue', 'Property ''ncl'' must be smaller than ''nco''.')
			end
			obj.ncl=val;
			
			% reset the NPC structure
			obj.resetNPC;
		end
		
			
		%------------------------------------------------------------------
		% set.wavelength
		%------------------------------------------------------------------
		function set.wavelength(obj, val)
			% controls and set the wavelength of interest
			
			if ~isnumeric(val) || any(val(:)<=0)
				error('SIF:wrongvalue', 'Property ''wavelength'' must be a vector of positive values.')
			end
			obj.wavelength=val(:)';
		end
		
			
		%------------------------------------------------------------------
		% set.birefringence
		%------------------------------------------------------------------
		function set.birefringence(obj, val)
			% controls and set the birefringence of the fiber
			
			if ~isscalar(val) || val(:)<0
				error('SIF:wrongvalue', 'Property ''birefringence'' must be a nonnegative values.')
			end
			obj.birefringence=val;
		end
		
			
		%------------------------------------------------------------------
		% set.ellipticity
		%------------------------------------------------------------------
		function set.ellipticity(obj, val)
			% Controls and set the ellipticity of the fiber.
			% 'deltaRatio' is the ratio between the maximum core radius 
			% variation and the core radius itself.
			% 'compoenents', indicates which field components to use (can
			% be 'all', 'transverse', or 'longitudinal')
			
			if ~isstruct(val) || ~all(ismember(fieldnames(val), {'deltaRatio', 'components'}))
				error('SIF:syntax', 'Syntax error.')
			end
			
			fn=fieldnames(val);
			for n=1:length(fn)
				switch fn{n}
					case 'deltaRatio'
						if ~isempty(val.deltaRatio) && (~isscalar(val.deltaRatio) || val.deltaRatio<0)
							error('SIF:wrongvalue', 'The parameter ''delta'' must be a nonnegative scalar.')
						end
						obj.ellipticity.deltaRatio=val.deltaRatio;
					case 'components'
						if  ~ischar(val.components) || ~ismember(val.components, {'all', 'transverse', 'longitudinal'})
							error('SIF:wrongvalue', 'Unknown component ''%s''.', val.components)
						end
						obj.ellipticity.components=val.components;
				end
			end
		end
		
			
		%------------------------------------------------------------------
		% set.bending
		%------------------------------------------------------------------
		function set.bending(obj, val)
			% Controls and set the bending of the fiber..
			% 'radius' is the bending radius [m].
			% 'components' indicates which field components to use (can
			% be 'all', 'transverse', or 'longitudinal')
            			
			if ~isstruct(val) || ~all(ismember(fieldnames(val), {'radius', 'components'}))
				error('SIF:syntax', 'Syntax error.')
			end
			
			fn=fieldnames(val);
			for n=1:length(fn)
				switch fn{n}
					case 'radius'
						if ~isscalar(val.radius) || val.radius<0
							error('SIF:wrongvalue', 'The parameter ''bending.radius'' must be a nonnegative scalar.')
						end
						obj.bending.radius=val.radius;
					case 'components'
						if  ~ischar(val.components) || ~ismember(val.components, {'all', 'transverse', 'longitudinal'})
							error('SIF:wrongvalue', 'Unknown component ''%s''.', val.components)
						end
						obj.bending.components=val.components;
				end
			end
		end
		
			
		%------------------------------------------------------------------
		% set.twist
		%------------------------------------------------------------------
		function set.twist(obj, val)
			% Controls and set the twist of the fiber.
			% 'rate' is the twist rate [rad/m] applied to the fiber.
			% 'g' is the elasto-optic coefficient.
			
			if ~isstruct(val) || ~all(ismember(fieldnames(val), {'rate', 'g'}))
				error('SIF:syntax', 'Syntax error.')
			end
			
			fn=fieldnames(val);
			for n=1:length(fn)
				switch fn{n}
					case 'rate'
						if ~isscalar(val.rate)
							error('SIF:wrongvalue', 'The parameter ''rate'' must be a scalar.')
						end
						obj.twist.rate=val.rate;
					case 'g'
						if  ~isscalar(val.g) 
							error('SIF:wrongvalue', 'The parameter ''g'' must be a scalar (typically about 0.146).')
						end
						obj.twist.g=val.g;
				end
			end
		end


		%------------------------------------------------------------------
		% get.NA
		%------------------------------------------------------------------
		function val=get.NA(obj)
			% calculate the numerical aperture
			
			if isempty(obj.nco) || isempty(obj.ncl)
				error('SIF:missingData', 'Both or one of the refractive indeces have not been specified.')
			end
			val=sqrt(obj.nco.^2 - obj.ncl.^2);			
		end


		%------------------------------------------------------------------
		% get.frequency
		%------------------------------------------------------------------
		function val=get.frequency(obj)
			% calculate the frequency in THz
			
			if isempty(obj.wavelength)
				error('SIF:missingData', 'Wavelengths have not been specified.')
			end
			val=SIF.c0./obj.wavelength;			
		end
				
		
		%------------------------------------------------------------------
		% get.normFrequency
		%------------------------------------------------------------------
		function val=get.normFrequency(obj)
			% calculate the normilized frequency
			
			if isempty(obj.wavelength) || isempty(obj.radius)
				error('SIF:missingData', 'Wavelengths and/or radius have not been specified.')
			end
			val=2*pi*obj.radius*obj.NA./obj.wavelength;	
		end			
		
	end
	
	
	%
	%______________________________________________________________________
	%______________________________________________________________________
	%
	%

	methods

		
		%------------------------------------------------------------------
		% obj.cutoffFreq
		%------------------------------------------------------------------
		function varargout=cutoffFreq(obj, varargin)
					
			% See method 'normCutoffFreq' for syntax and help. This is the
			% same but return actual frequencies.
			
			% call 'normCutoffFreq' 
			[cof, modes]=obj.normCutoffFreq(varargin{:});
			
			% de-normalize frequencies
			varargout{1}=SIF.c0*cof/(2*pi*obj.radius*obj.NA);
			
			if nargout>1
				varargout{2}=modes;
			end
		end
			

		%------------------------------------------------------------------
		% obj.propagationConst(modes)
		%------------------------------------------------------------------
		function beta=propagationConst(obj, varargin)
			
			% beta=propagationConst(obj, varargin)
			%
			% Calculates the propagation constants for the given mode. See
			% method 'normPropagationConst' for information about the
			% syntax.
            %
            % See also SIF.normPropagationConst
			
			b=obj.normPropagationConst(varargin{:});
			
			% de-normalize
			omega=2*pi*obj.frequency;
			kco2=omega.^2 * SIF.mu0 * SIF.epsilon0 * obj.nco^2;
			kcl2=omega.^2 * SIF.mu0 * SIF.epsilon0 * obj.ncl^2;
			
			beta=sqrt(b.*repmat(kco2-kcl2, size(b,1), 1) + ...
				repmat(kcl2, size(b,1), 1));
		end
	
		
		%------------------------------------------------------------------
		% obj.couplingMatrix
		%------------------------------------------------------------------
		function varargout=couplingMatrixLP(obj, varargin)
			
			% Calculates the couplingMatrix for the given LP modes and the
			% given effect, in the frequency range specified by obj. The
			% coupling matrix K is defined so that da/dz = -j K a;
			% accordingly, K is hermitian.
			%
			%   [K, indx]=obj.couplingMatrixLP(modes, perturbation)
			%   modes=obj.couplingMatrixLP(perturbation)
			%
			% In the first syntax:
			% 'modes' is the usual set of modes. 
			% 'perturbation' is a string or cell array of string with the 
			% perturbations to be considered in the calculation of K.
			% 
			% The function returns the coupling matrix K of size (MxMxnf),
			% where M is the number of modes considering also the
			% *degeneracies*, and nf is the number of frequencies. The
			% matrix is calculated only for LP modes, and degenerate modes 
			% are sorted according to the sequence
			%   (cos,x) (cos,y) (sin,x) (sin,y)
			% 'indx' is a Nx2 matrix with the indeces of the non-zero
			% entries of K. Only the upper triangular part of the matrix is
			% considered; remember that the matrix K is hermitian.
			%
			% Note that for each perturbation the coupling coefficients are
			% defined only for a set of modes. To know which modes, use the
			% second syntax for each perturbation.
			%
			% To transform the calculated coupling matrix to the base of
			% hybrid modes, you can calculate K_hy = A*K*A', where K is the
			% matrix calculated here, and A is the matrix returned by
			% method 'FiberModes.lp2hybrid'.
			%
			
			%--------------------------------------------------------------
			% check the input arguments
			narg=length(varargin);
			if narg>2
				error('SIF:syntax', 'Syntax error.')
			end
				
			modes=FiberModes;
			perturb={};
			
			for n=1:narg		
				if isa(varargin{n}, 'FiberModes')
					
					% these are the mode to be used for calculation
					modes=varargin{n};
				
				elseif ischar(varargin{n}) || iscellstr(varargin{n})
				
					% these can be either modes or perturbations
					if all(ismember(varargin{n}, obj.perturbations))
						
						% yes it's a list of perturbations
						perturb=varargin{n};
						
					else
						
						% could other modes
						mo=FiberModes(varargin{n});
						if ~isempty(modes)
							error('SIF:syntax', 'Modes have been specified twice.')
						end
						modes=mo;
					end
				end
			end
			if isempty(perturb)
				% set a default value
				perturb=obj.perturbations;
			elseif ~iscell(perturb)
				perturb={perturb};
			end
			
			
			%--------------------------------------------------------------
			% now acts
			if isempty(modes) 
				
				if length(perturb)==1
					% the user wants to know for which modes can be the given
					% perturbation be calculated
					mi=obj.(sprintf('calc_%s', perturb{1}))();
					
					% converts the mode indeces in a mode object
					str='';
					for n=1:size(mi,1)
						str=sprintf('%s LP(%d,%d)', str, mi(n,1), mi(n,2));
					end
					varargout{1}=FiberModes(str);
				else
					error('SIF:syntax', 'The perturbation has not been specified.')
				end
				
			else
				
				% the user wants to calculate the coupling matrix
				
				% looks for the LP modes
				nm=modes.nmodes;
				lp=find(modes.modes(:,1)==5);
				hy=setdiff(1:nm, lp);
				if ~isempty(hy)
					error('SIF:notImplemented', 'Sorry hybrid modes are not implemented.')
				end
				
				% total number of modes (including degeneracies)
				nt=sum((modes.modes(lp,2)>0)*4 + (modes.modes(lp,2)==0)*2);
				
				% number of frequencies
				nf=length(obj.wavelength);
			
				% allocates the output matrix 
				K=zeros(nt,nt,nf);
				
				% and the indeces to the nonzero elements
				indx=zeros(0,2);
				
				% cycle over the perturbations
				for p=1:length(perturb)
					
					% check is the perturbation parameters have been
					% defined
					if isstruct(obj.(perturb{p}))
						fn=fieldnames(obj.(perturb{p}));
						for q=1:length(fn)
							if isempty(obj.(perturb{p}).(fn{q}))
								error('SIF:missingData', ...
									'The parameters of perturbation ''%s'' are missing.', ...
									perturb{p})
							end
						end
					else
						if isempty(obj.(perturb{p}))
							error('SIF:missingData', ...
									'The parameters of perturbation ''%s'' are missing.', ...
									perturb{p})
						end
					end
					
					% calculate the coupling matrix
					[Kp, i]=obj.(sprintf('calc_%s', perturb{p}))(modes);
					
					% the total matrix
					K=K+Kp;
					
					% records the nonzero elements
					indx=[indx; i];
				end
				
				varargout{1}=K;
				if nargout>1
					% removes duplicates from indx
					if ~isempty(indx)
						hash=indx(:,1)*(max(indx(:,2))+1) + indx(:,2);
						[~, j]=unique(hash);
						indx=indx(j,:);
					end
					varargout{2}=indx;
				end
				
			end		
						
		end
			
		
		%------------------------------------------------------------------
		% obj.J(modes, r)
		%------------------------------------------------------------------
		function val=J(obj, modes, r)
			
			%  val=obj.J(modes, r)
			%
			% Returns the values of the radial term of modes specified by 
			% 'modes' at the coordinate given by vector r for the frequency 
			% specfied by obj.frequency.
			% 'val' is a matrix of size (length(obj.frequency) x length(r));
			% if more than a mode is specified returns a cell array of such
			% matrices.
			
			% a check
			if ischar(modes)
				modes=FiberModes(modes);
			end
			
			% looks for the LP modes
			nm=modes.nmodes;
			lp=find(modes.modes(:,1)==5);
			hy=setdiff(1:nm, lp);
			if ~isempty(hy)
				error('SIF:notImplemented', 'Sorry hybrid modes are not implemented yet.')
			end
			
			% a check
			if isempty(obj.radius)
				error('SIF:missingData', 'The radius has not been specified.')
			end
			
			% determines the radial coordinate inside and outside the core
			r=r(:);
			nr=numel(r);
			inco=find(r<=obj.radius);
			outco=find(r>obj.radius);
			
			% angular frequency
			nf=length(obj.frequency);
			omega=2*pi*obj.frequency;

			% allocate the output
			val=cell(1,nm);
			for m=1:nm
				val{m}=zeros(nr,nf);
			end
			
			%-------------------------------------
			% calculate the propagation constants
			beta=obj.propagationConst(modes);
			
			%-------------------------------------
			% calculate the terms inside the core
			if ~isempty(inco)
				
				% cycle over modes
				for m=1:nm
					
					% the u parameter
					u=sqrt(omega.^2 * SIF.mu0 * SIF.epsilon0 * obj.nco^2 - beta(m,:).^2);
				
					% calculate J
					[U,R]=meshgrid(u,r(inco));
					val{m}(inco,:)=besselj(modes.modes(m,2), U.*R)./besselj(modes.modes(m,2), U.*obj.radius);
				end
			end
			
			%-------------------------------------
			% calculate the terms outside the core
			if ~isempty(outco)
				
				% cycle over modes
				for m=1:nm
					
					% the w parameter
					w=sqrt(beta(m,:).^2 - omega.^2 * SIF.mu0 * SIF.epsilon0 * obj.ncl^2);
				
					% calculate J
					[W,R]=meshgrid(w,r(outco));
					val{m}(outco,:)=besselk(modes.modes(m,2), W.*R)./besselk(modes.modes(m,2), W.*obj.radius);
				end
			end
			
			if length(val)==1
				val=val{1};
			end
			
		end
			
		
		%------------------------------------------------------------------
		% obj.dJ(modes, r)
		%------------------------------------------------------------------
		function val=dJ(obj, modes, r)
			
			% Returns the values of the derivative of the radial term of 
			% modes specified by 'modes' at the coordinate given by vector 
			% r for the frequency specfied by obj.frequency.
			% 'val' is matrix of size (length(obj.frequency) x length(r));
			% if more than a mode is specified returns a cell array of such
			% matrices.
			
			% a check
			if ischar(modes)
				modes=FiberModes(modes);
			end
			
			% looks for the LP modes
			nm=modes.nmodes;
			lp=find(modes.modes(:,1)==5);
			hy=setdiff(1:nm, lp);
			if ~isempty(hy)
				error('SIF:notImplemented', 'Sorry hybrid modes are not implemented yet.')
			end
			
			% a check
			if isempty(obj.radius)
				error('SIF:missingData', 'The radius has not been specified.')
			end
			
			% determines the radial coordinate inside and outside the core
			r=r(:);
			nr=numel(r);
			inco=find(r<=obj.radius);
			outco=find(r>obj.radius);
			
			% angular frequency
			nf=length(obj.frequency);
			omega=2*pi*obj.frequency;

			% allocate the output
			val=cell(1,nm);
			for m=1:nm
				val{m}=zeros(nr,nf);
			end
			
			%-------------------------------------
			% calculate the propagation constants
			beta=obj.propagationConst(modes);
			
			%-------------------------------------
			% calculate the terms inside the core
			if ~isempty(inco)
				
				% cycle over modes
				for m=1:nm
					
					% the u parameter
					u=sqrt(omega.^2 * SIF.mu0 * SIF.epsilon0 * obj.nco^2 - beta(m,:).^2);
				
					% azimuthal order
					n=modes.modes(m,2);
					
					% calculate dJ
					[U,R]=meshgrid(u,r(inco));
					val{m}(inco,:)=U.*(besselj(n-1, U.*R) - besselj(n+1,U.*R))./(2*besselj(n, U.*obj.radius));
				end
			end
			
			%-------------------------------------
			% calculate the terms outside the core
			if ~isempty(outco)
				
				% cycle over modes
				for m=1:nm
					
					% the w parameter
					w=sqrt(beta(m,:).^2 - omega.^2 * SIF.mu0 * SIF.epsilon0 * obj.ncl^2);
				
					% azimuthal order
					n=modes.modes(m,2);
				
					% calculate dJ
					[W,R]=meshgrid(w,r(outco));
					val{m}(outco,:)=-W.*(besselk(n-1, W.*R) + besselk(n+1, W.*R))./(2*besselk(n, W.*obj.radius));
				end
			end
			
			if length(val)==1
				val=val{1};
			end
			
		end

				
		%------------------------------------------------------------------
		% obj.Q(modes)
		%------------------------------------------------------------------
		function val=Q(obj, modes)

			% val=obj.Q(modes)
			%
			% Calculate the normalization factors
			%
			%   Integral_0^inf r*J(r)^2 dr
			%
			% Returns a matrix with one element for each mode and each
			% frequency.
			
			
			% a check
			if ischar(modes)
				modes=FiberModes(modes);
			end
			
			% number of modes
			nm=modes.nmodes;
			
			% number of frequency
			nf=length(obj.wavelength);

			% allocate the output
			val=zeros(nm, nf);

			% defines a range where to evaluate the integrals
			dr=obj.radius/100;
			r=0:dr:obj.radius*(5+max(modes.modes(:,3)));
			
			% calculate J(r)
			JJ=obj.J(modes, r);
			
			% evaluates the integrals
			for m=1:nm
				if nm==1
					val(m,:)=trapz(r, repmat(r',1,nf) .* JJ.^2);
				else
					val(m,:)=trapz(r, repmat(r',1,nf) .* JJ{m}.^2);
				end
            end
        end
	end
	
	%
	%______________________________________________________________________
	%______________________________________________________________________
	%
	%
	
	
	% prototypes

	methods 
			
		% calculate the normalized cut off frequencies
		[cof, modes]=normCutoffFreq(obj, varargin);

		% calculate the normalized propagation constant
		[cp, modes, cof]=normPropagationConst(obj, varargin); 
	
	end
	
	
	%
	%______________________________________________________________________
	%______________________________________________________________________
	%
	%

	methods (Access = private)

		
		%------------------------------------------------------------------
		% calcNPC (prototype)
		% calculate the NPC for a specific mode from its cutoff upto a
		% given frequency
		%------------------------------------------------------------------
		calcNPC(obj, mode, cof, vmax)
			
		
		%------------------------------------------------------------------
		% functions to calculate the coupling matrix (prototypes)
		varargout=calc_birefringence(obj, varargin)
		varargout=calc_ellipticity(obj, varargin)
		varargout=calc_bending(obj, varargin)
		varargout=calc_twist(obj, varargin)
		
		
		%------------------------------------------------------------------
		% loadNPC
		%------------------------------------------------------------------
		function loadNPC(obj)

			% load the current NPC, if available
			% *must* be called only during initialization!!!
			
			fid=fopen(fullfile(obj.dirNPC, obj.fileNPC), 'r');
			if fid>1
				% yes we have a file
				fclose(fid);
				
				% load the data in the varaible 'data'
				data=load(fullfile(obj.dirNPC, obj.fileNPC));
			
				for m=1:length(data.modes)
					% add the mode
					obj.NPC.modes.add(data.modes{m});
					% add the cof
					obj.NPC.cof(m)=data.cof(m);
					% add the v
					obj.NPC.v{m}=data.v{m};
					% add the b
					obj.NPC.b{m}=data.b{m};
				end
			end
		end
			
			
		%------------------------------------------------------------------
		% saveNPC
		%------------------------------------------------------------------
		function saveNPC(obj)

			% save the current NPC; save only the parts relative to LP
			% and TE modes (which do not depend on the refractive indeces) 
			% and only if the current NPC are calculated on a larger 
			% frequency range
			
			% if the current NPC is empty, there is nothing to do
			if isempty(obj.NPC.modes)
				return
			end
			
			% load stored data (if any)
			fid=fopen(fullfile(obj.dirNPC, obj.fileNPC), 'r');
			if fid>1
				% yes we have a file
				fclose(fid);
				% load the data in the variable 'data'
				data=load(fullfile(obj.dirNPC, obj.fileNPC));
			else
				% initialize the data
				data=struct('modes', {{}}, 'cof', [], ...
					'b', {{}}, 'v', {{}});
			end
			
			% boolean mask of the current NPC that could be saved
			mask=(obj.NPC.modes.modes(:,1)==3 | obj.NPC.modes.modes(:,1)==5);

			% cycle over the saved data
			for m=1:length(data.modes)
				% let's see if this mode should be saved
				j=obj.NPC.modes.find(FiberModes(data.modes{m}));
				
				if max(obj.NPC.v{j}) > max(data.v{m})
					% yes we must update the file
					data.v{m}=obj.NPC.v{j};
					data.b{m}=obj.NPC.b{j};
					data.cof(m)=obj.NPC.cof(j);
					% the last one should not change, but you never know...
				end
				
				% mark the mode as "done"
				mask(j)=false;
			end
			
			% cycle over the local NPC modes that have not been saved so
			% far; this means that they were not in the saved data
			for m=find(mask')
				% current mode
				cm=obj.NPC.modes(m);
				data.modes{end+1}=cm.string;
				data.cof(end+1)=obj.NPC.cof(m);
				data.v{end+1}=obj.NPC.v{m};
				data.b{end+1}=obj.NPC.b{m};
			end
			
			% save the update NPC file
			save(fullfile(obj.dirNPC, obj.fileNPC), '-struct', 'data')
			
		end
			
			
		%------------------------------------------------------------------
		% resetNPC
		%------------------------------------------------------------------
		function resetNPC(obj, mode)
			
			% remove every entry in obj.NPC that refers to modes that are
			% not LP or TE, or to the mode specified in 'mode'
			
			if ~isempty(obj.NPC.modes)
				
				if nargin==1
					% no specific mode has been given
					% removes all the non LP and non TE
					
					% mode types
					mT=obj.NPC.modes.modes(:,1);
					
					% items to be preserved (TE and LP)
					P=find(mT==4 | mT==5);
					
					% removes the entries
					if isempty(P)
						% just reset the NPC
						obj.NPC.modes=[];
						obj.NPC.cof=[];
						obj.NPC.v={{}};
						obj.NPC.b={{}};
					else
						% keeps the good data
						obj.NPC.modes=obj.NPC.modes(P);
						obj.NPC.cof=obj.NPC.cof(P);
						obj.NPC.v=obj.NPC.v(P);
						obj.NPC.b=obj.NPC.b(P);
					end
					
				else
					
					% remove the requested mode
					if ~isa(mode, 'FiberModes') || mode.nmodes~=1
						error('SIF:syntax', 'Only one mode at a time can be removed from NPC.')
					end
					
					tm=obj.NPC.modes.find(mode);
					if ~isempty(tm)
						% items to be preserved 
						P=setdiff(1:obj.NPC.modes.nmodes, tm);
						% keeps the good data
						obj.NPC.modes=obj.NPC.modes(P);
						obj.NPC.cof=obj.NPC.cof(P);
						obj.NPC.v=obj.NPC.v(P);
						obj.NPC.b=obj.NPC.b(P);
					end
				end
			end
		end
		
	end

	
	%
	%______________________________________________________________________
	%______________________________________________________________________
	%
	%


	methods (Static, Access=private)
			
			
		%------------------------------------------------------------------
		% matrixIndeces
		%------------------------------------------------------------------
		function [J, ntotmodes]=matrixIndeces(modes, modeIndeces)
		
			%
			%  [J, ntotmodes]=SIF.matrixIndeces(modes, modeIndeces)
			% 
			% Given a set of LP modes (as FiberModes object) and a list of 
			% LP mode indeces (as a Nx2 matrix), returns:
			%
			% - J: the list (1xM) of indeces such that k-th mode of 'modes'
			%   corresponds to the J(k)-th mode of 'modeIndeces'. Note that
			%   the indeces includes also the degenerate mode!!
			% - ntotmodes: the total number of modes listed in 'modes'
			%   (including the degenerate ones). 
			%
			% If not all the modes in 'modes' are listed in 'modeIndeces',
			% an error is raised. 
			%
			% In calculating the indeces, also degenerate modes are 
			% considered and they are sorted with the "usual" order 
			%    (cos,x), (cos,y), (sin,x), (sin,y)
			%
			
			% convert 'modeIndeces' in a FiberModes object
			modestr='';
			for n=1:size(modeIndeces,1)
				modestr=sprintf('%s LP(%d,%d)', modestr, ...
					modeIndeces(n,1), modeIndeces(n,2));
			end
			themodes=FiberModes(modestr);
			
			% check if all the requested modes are among the "known" ones
			imodes=zeros(1,modes.nmodes);
			for n=1:modes.nmodes
				j=themodes.find(modes(n));
				if isempty(j)
					mo=modes(n);
					error('SIF:missingData', 'The coupling matrix is not defined for the mode %s.', ...
						mo.string)
				end
				% stores the position of modes(n) in modeIndeces
				imodes(n)=j;
			end
			
			% count the total number of modes in 'modes'
			ntotmodes=sum(2*(modes.modes(:,2)==0) + 4*(modes.modes(:,2)>0));
			
			% cumulative number of modes in 'modeIndeces'
			cummodes=cumsum(2*(modeIndeces(:,1)==0) + 4*(modeIndeces(:,1)>0));
			cummodes=[0; cummodes(1:end-1)];
			
			% determines J
			J=zeros(1, ntotmodes);
			c=0;
			for n=1:modes.nmodes
				j=1:2*(1 + (modes.modes(n,2)~=0));
				J(c+j)=cummodes(imodes(n))+j;
				c=c+j(end);
			end
			
		end	
	
	end
	
end