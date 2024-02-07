%
% This class implement a simple container to handle fiber mode indeces; 
% entries in the container are unique.
% 

classdef FiberModes < handle
	
	
	properties (SetAccess = private)
		
		% array of modes 
		% - column 1 stores the index of the mode type (from 1 to 5, see 
		%   'labels' below)
		% - column 2 stores the azimuthal index of the mode
		% - column 3 stores the radial index of the mode
		modes=zeros(0,3);
		
		% number of modes without counting degeneracies
		% (this is also the number of rows of modes)
		nmodes=0;
		
	end
	
	properties (Dependent = true)
		
		% number of modes counting degeneracies
		ndegmodes;
		
	end
	
	properties (Access = private)
		
		% array of modes, the real one!
		themodes=zeros(0,3);
		
	end
	
	properties (Constant)
		
		% mode labels in the order as they are numbered
		labels={'HE', 'EH', 'TM', 'TE', 'LP'};
		
	end
	
	
	methods
		
		%------------------------------------------------------------------
		% constructor
		%------------------------------------------------------------------
		function obj=FiberModes(varargin)
			
			% Possible syntaxes:
			% FiberModes()
			% FiberModes(fibermodes)
			% FiberModes(type, indx, type, indx, 'te(0,2) lp(4,4)', ...)
			
			if ~isempty(varargin)
				% add the modes
				obj.add(varargin{:})
			end
			
		end
		
		
		%------------------------------------------------------------------
		% isempty
		%------------------------------------------------------------------
		function tf=isempty(obj)
			
			if obj.nmodes==0
				tf=true;
			else
				tf=false;
			end
		end	
		
		
		%------------------------------------------------------------------
		% add
		%------------------------------------------------------------------
		function add(obj, varargin)

			% Syntax:
			%   obj.add(type, indx, type, indx, ...)
			%   obj.add(fibermodes)
			
			if ~isempty(varargin)

				if isa(varargin{1}, 'FiberModes')
					% we are adding modes from an object
					
					% number of new modes
					nm=varargin{1}.nmodes;
					
					% new modes
					newmodes=varargin{1}.modes;
					
				else
					% we are adding modes "from the command line"
					
					% estimated number of new modes
					enm=length(varargin);
					
					% new modes
					newmodes=zeros(enm, 3);
					
					% actual number of modes
					nm=0;
					
					% regular expression that defines the modes
					re='(te|tm|he|eh|lp)\(([0-9]+),([0-9]+)\)';
					
					% cycle on the input parameters
					n=1;
					while n<=length(varargin)
						
						arg=varargin{n};
						
						if ischar(arg)
							% is it an expression?
							tkns=regexpi(arg, re, 'tokens');
							if ~isempty(tkns)
								
								% yes we have an expression
								tnm=length(tkns);
								
								% type-strings and indeces
								typestr=cell(1,tnm);
								indeces=zeros(tnm,2);
								
								for k=1:tnm
									typestr{k}=tkns{k}{1};
									indeces(k,:)=[str2double(tkns{k}{2}) str2double(tkns{k}{3})];
								end
								
								% update for the next argument
								n=n+1;
								
							else
								
								% probably, we have a type followed by a
								% vector of indeces
								typestr=varargin(n);
								if length(varargin)<n+1
									error('FiberModes:syntax', 'Syntax error')
								end
								indeces=varargin{n+1};
								indeces=indeces(:)';   % we really want a row
								if ~isnumeric(indeces) || numel(indeces)~=2
									error('FiberModes:syntax', 'The indeces must be 2-dimensional vectors.')
								end
						
								% update for the next argument
								n=n+2;
								
							end
							
						else
							
							% this must an error
							error('FiberModes:syntax', 'Syntax error.')
							
						end
						
						% adds the modes just found
						for k=1:length(typestr)
						
							% increment the mode counter
							nm=nm+1;
							
							% determine the type
							type=find(strcmpi(typestr{k}, FiberModes.labels));
							if isempty(type)
								error('FiberModes:syntax', 'Unknown mode ''%s''.', typestr)
							end
							newmodes(nm,1)=type;
							
							% determine the index
							if indeces(k,1)<0 || rem(indeces(k,1),1)~=0
								error('FiberModes:wrongIndex', 'The azimuth order must be a nonnegative integer.')
							end
							if indeces(k,2)<1 || rem(indeces(k,2),1)~=0
								error('FiberModes:wrongIndex', 'The radial order must be a positive integer.')
							end
							if (type==3 || type==4) && indeces(k,1)~=0
								error('FiberModes:wrongIndex', 'TE and TM modes have necessarily azimuth order equal to 0.')
							end
							if (type==1 || type==2) && indeces(k,1)==0
								error('FiberModes:wrongIndex', 'HE and EH modes have necessarily azimuth order larger than 1.')
							end
							
							% finally
							newmodes(nm,2:3)=indeces(k,:);
							
						end
						
					end
						
				end
				
				% merges the new modes to the previous ones, removing
				% duplicates
				
				% allocates enough space (if needed)
				if size(obj.themodes,1)-obj.nmodes < nm
					% we potentially need more space
					obj.themodes=[obj.themodes; zeros(nm, 3)];
				end
				
				% cycle over the newmodes
				for n=1:nm
					
					% check if the mode is already in the set
					addit=true;
					for k=1:obj.nmodes
						if all(obj.themodes(k,:)==newmodes(n,:))
							% its a duplicate
							addit=false;
							break
						end
					end
					
					if addit
						% it's not a duplicate, let's add it
						obj.nmodes=obj.nmodes+1;
						obj.themodes(obj.nmodes,:)=newmodes(n,:);
					end
				end			
			end
		end
		
		
		%------------------------------------------------------------------
		% get.modes
		%------------------------------------------------------------------
		function val=get.modes(obj)
			
			% returns only the useful portion of themodes
			val=obj.themodes(1:obj.nmodes,:);
		end
		
		
		%------------------------------------------------------------------
		% get.ndegmodes
		%------------------------------------------------------------------
		function val=get.ndegmodes(obj)
			
			% returns the number of modes including degeneracies
			val=sum((obj.modes(:,1)==3 | obj.modes(:,1)==4) + ...
				2*(obj.modes(:,1)==5 & obj.modes(:,2)==0) + ...
				4*(obj.modes(:,1)==5 & obj.modes(:,2)>0) + ...
				2*(obj.modes(:,1)==1 | obj.modes(:,1)==2));
			
		end
		
		
		%------------------------------------------------------------------
		% degenerate
		%------------------------------------------------------------------
		function I=degenerate(obj)
			
			%
			% I=obj.degenerate
			%
			% Returns a vector of size (1xobj.ndegmodes) such that
			% obj(I(k)) is the mode referring to the k-th degenerate mode.
			%
			
			I=zeros(1,obj.ndegmodes);
			j=0;
			for n=1:obj.nmodes
				if obj.modes(n,1)==3 || obj.modes(n,1)==4
					i=1;
				elseif obj.modes(n,1)==1 || obj.modes(n,1)==2 || ...
						(obj.modes(n,1)==5 && obj.modes(n,2)==0)
					i=[1 2];
				else
					i=1:4;
				end
				I(j+i)=n;
				j=j+length(i);
			end
		end
		
		
		%------------------------------------------------------------------
		% subsref
		%------------------------------------------------------------------
		function varargout=subsref(obj, S)
			
			% implements obj(1), obj(1:3), ...
			
			% only () is considered
			if length(S)>1 || ~strcmp(S.type, '()')
				
				% calls the builtin subrefs
				[varargout{1:nargout}]=builtin('subsref', obj, S);

			else
				
				% only 1D subrefs is performed
				if length(S.subs)~=1
					error('FiberModes:syntax', 'Only 1-dimensional subsref are supported.')
				end
				
				% check the indeces
				if any(S.subs{1}<1) || any(S.subs{1}>obj.nmodes)
					error('fiberModes:indeces', 'Indeces out of range.')
				end
				
				% create the list of wanted modes
				modes='';
				for n=1:numel(S.subs{1})
					j=S.subs{1}(n);
					modes=sprintf('%s %s(%d,%d)', modes, ...
						FiberModes.labels{obj.modes(j,1)}, ...
						obj.modes(j,2), obj.modes(j,3));
				end
				
				% create the new object
				varargout{1}=FiberModes(modes);
			
			end
			
		end
		
		
		%------------------------------------------------------------------
		% find
		%------------------------------------------------------------------
		function val=find(obj, varargin)
			
			% if 'varargin' specifies a mode that is already in obj,
			% returns its position, otherwise returns an empty matrix
			
			% create the searched mode
			if isa(varargin{1}, 'FiberModes')
				if length(varargin)>1
					error('FiberModes:syntax', 'Too many input arguments.')
				end
				fm=varargin{1};
			else
				fm=FiberModes(varargin{:});
			end
			
			% search the mode
			if fm.nmodes~=1
				error('fiberModes:syntax', 'You can find one and only one mode at a time.')
			end
			val=[];
			for n=1:obj.nmodes
				if all(obj.modes(n,:)==fm.modes)
					val=n;
					break
				end
			end
		end
		
		
		%------------------------------------------------------------------
		% sort
		%------------------------------------------------------------------
		function i=sort(obj)
			
			% Sort modes by type, then by azimuth order and finally by
			% radial order.
			% Returns the indeces that perform the sort 
			
			% create an index
			b=max(obj.modes(:))+1;
			indx=obj.modes(:,1)*b^2 + obj.modes(:,2)*b + obj.modes(:,3);
			
			% sorts by the index
			[~, i]=sort(indx);
			obj.themodes(1:obj.nmodes,:)=obj.themodes(i,:);
			
		end
		
		
		%------------------------------------------------------------------
		% string
		%------------------------------------------------------------------
		function str=string(obj)
			
			% returns a string or cell array of strings with the name of
			% the modes.
			
			str=cell(1,obj.nmodes);
			for n=1:obj.nmodes
				str{n}=sprintf('%s(%d,%d)', ...
					FiberModes.labels{obj.modes(n,1)}, ...
					obj.modes(n,2), obj.modes(n,3));
			end
			if n==1
				str=str{1};
			end
		end
				
	
		%------------------------------------------------------------------
		% disp
		%------------------------------------------------------------------
		function disp(obj)
			
			fprintf('Number of modes: %d\n', obj.nmodes)
			fprintf('Number of modes including degeneracies: %d\n\n', obj.ndegmodes)
			for n=1:obj.nmodes
				fprintf('%d. %s(%d,%d)\n', n, FiberModes.labels{obj.modes(n,1)}, ...
					obj.modes(n,2), obj.modes(n,3));
			end
			fprintf('\n');
			
		end
		
				
		
		%------------------------------------------------------------------
		% [hybrid, A]=obj.lp2hybrid
		%------------------------------------------------------------------
		function [hybrid, A]=lp2hybrid(obj)
			
			% 
			% [hybrid, A]=lp2hybrid(obj)
			%
			% Given a list of LP modes (as an FiberModes object) returns:
			% 'hybrid' a FiberModes object with the list of the hybrid
			%          modes that are approximated by the corresponding LP
			%          modes. Both objects do not explicitly list the
			%          degenerate modes, as a consequence, 'hybrid' has in
			%          general more elements than 'obj'. Quasi-degenerate
			%          modes are listed with the order HE, EH, TM and TE.
			% 'A'      is a matrix such that H=A*L, where L is the vector
			%          of coefficients of the LP modes and H is the vector
			%          of coefficients of the corresponding hybrid modes.
			%          This matrix includes all degenerate modes, and
			%          assumes that modes are ordered as follows:
			%            LP(e,x) LP(e,y) LP(o,x) LP(o,y)
			%            HE(e) HE(o) EH(e) EH(o)
			%          where 'e' stands for 'even' (i.e. cosine) and 'o'
			%          stands for 'odd' (i.e. sine), and EH(e) and EH(o)
			%          are, respectively, the TM and TE modes when the
			%          azimuth order is 0.
			%
			
			% check if all modes in 'obj' are actually LP modes
			if ~all(obj.modes(:,1)==5)
				error('FiberModes:wrongmodes', 'Input modes must be all LP.')
			end
			
			% builds the list of hybrid modes
			str='';
			for n=1:obj.nmodes
				switch obj.modes(n,2)
					case 0
						% LP(0,p) modes are made of HE modes
						str=sprintf('%s he(%d,%d)', str, ...
							obj.modes(n,2)+1, obj.modes(n,3));
					case 1
						% LP(1,p) modes are made of HE, TM and TE modes
						str=sprintf('%s he(2,%d) tm(0,%d) te(0,%d)', str, ...
							obj.modes(n,3), obj.modes(n,3), obj.modes(n,3));
					otherwise
						% LP(n,p) modes are made of HE and EH modes
						str=sprintf('%s he(%d,%d) eh(%d,%d)', str, ...
							obj.modes(n,2)+1, obj.modes(n,3), ...
							obj.modes(n,2)-1, obj.modes(n,3));
				end
			end
			hybrid=FiberModes(str);

			% build the trasformation matrix, if requested
			if nargout>1
				nm=sum((obj.modes(:,2)==0)*2 + (obj.modes(:,2)~=0)*4);
				A=zeros(nm,nm);
				j=0;
				for n=1:obj.nmodes
					if obj.modes(n,2)==0
						% LP(0,p) modes
						i=1:2;
						A(j+i, j+i)=eye(2);
						j=j+2;
					else
						% LP(n,p) modes
						i=1:4;
						A(j+i, j+i)=[  1  0  0 -1; 
							           0  1  1  0;
									  -1  0  0 -1; 
									   0  1 -1  0]/sqrt(2);
						j=j+4;
					end
				end
			end
		end
			

		
	end
	
end