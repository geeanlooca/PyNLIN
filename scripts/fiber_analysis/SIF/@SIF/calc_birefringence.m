%
% Calculates the coupling matrix for birefringent fiber.
%
%   [K, I]=obj.calc_birefringence(modes)
%   mi=obj.calc_birefringence()
%
% The first syntax returns the coupling matrix K for the specified modes;
% the matrix has size MxMxN, where M is the number of modes including the
% degenerate ones and N is the number of wavelengths specified in
% obj.wavelength. The matrix I (Px2) is a list of the non-zero elements of
% K; note that only the elements of the upper triangular portion are
% returned!
%
% The second syntax just returns the list of modes for which the coupling
% matrix can be calculated.
%
function varargout=calc_birefringence(obj, varargin)


% matrix of the mode incedes
modeIndeces=[0,1;1,1;2,1;0,2;3,1];

if isempty(varargin)
	% returns the known mode indeces
	varargout{1}=modeIndeces;

	% we've done 
	return
end


% we have to calculate the matrix
modes=varargin{1};

% check if all the modes are known
[indx, dims]=SIF.matrixIndeces(modes, modeIndeces);

% define the parameters
c=SIF.c0;
mu0=SIF.mu0;
lambda=obj.wavelength;
% delta epsilon = epsilon_x -epsilon_y = 2 epsilon_0 n_co (n_x - n_y)
deltaEpsilon=obj.birefringence*2*obj.nco/(mu0*c^2);

% define the functions
J=@(n,m,r) obj.J(sprintf('LP(%d,%d)', n, m), r);
dJ=@(n,m,r) obj.dJ(sprintf('LP(%d,%d)', n, m), r);
Q=@(n,m) obj.Q(sprintf('LP(%d,%d)', n, m));
beta=@(n,m) obj.propagationConst(sprintf('LP(%d,%d)', n, m));


% the perturbation matrix (just for the records)
% perturbation=[(1/2).*deltaEpsilon,0,0;0,(-1/2).*deltaEpsilon,0;0,0,0];

% normalization factor; this multiplies the coupling matrix
% WARNING: the normalization factor given by Mathematica must be multiplied
%          by sqrt(-1)! This is needed to make the resulting coupling
%          consistent with the definition given in method 'couplingMatrix'.
%          The expression reported below is already corrected by the factor
%          sqrt(-1).
norm_factor=-c.^2.*deltaEpsilon.*lambda.^(-2).*mu0.*pi.^2;


% define the coupling matrix as a cell array
% only the elements of the upper 'diagonal' are specified
% only non-zero elements are specified

% define the coupling matrix as a cell array
% only the elements of the upper 'diagonal' are specified
% only non-zero elements are specified
Kb=cell(16,16);

% the non-zero elements
Kb{1,1}=(-1).*beta(0,1).^(-1);
Kb{2,2}=beta(0,1).^(-1);
Kb{3,3}=(-1).*beta(1,1).^(-1);
Kb{4,4}=beta(1,1).^(-1);
Kb{5,5}=(-1).*beta(1,1).^(-1);
Kb{6,6}=beta(1,1).^(-1);
Kb{7,7}=(-1).*beta(2,1).^(-1);
Kb{8,8}=beta(2,1).^(-1);
Kb{9,9}=(-1).*beta(2,1).^(-1);
Kb{10,10}=beta(2,1).^(-1);
Kb{11,11}=(-1).*beta(0,2).^(-1);
Kb{12,12}=beta(0,2).^(-1);
Kb{13,13}=(-1).*beta(3,1).^(-1);
Kb{14,14}=beta(3,1).^(-1);
Kb{15,15}=(-1).*beta(3,1).^(-1);
Kb{16,16}=beta(3,1).^(-1);


%--------------------------------------------------------------------------
% calculate the coupling matrix
%--------------------------------------------------------------------------

nf=length(obj.wavelength);
CM=zeros(dims, dims, nf);
I=zeros(dims*(dims+1)/2, 2);

% cycle over matrix elements
c=0;
for n=1:dims
	for m=n:dims
		
		% true if the element is not zero
		nonzero=false;
		
		if ~isempty(Kb{indx(n),indx(m)})
			% there is something to calculate
			CM(n,m,:)=Kb{indx(n),indx(m)};
			nonzero=true;
		elseif ~isempty(Kb{indx(m),indx(n)})
			% we have to calculate the transpose conjugate
			CM(n,m,:)=conj(Kb{indx(m),indx(n)});
			nonzero=true;
		end
		
		% update the list of nonzero elements
		if nonzero
			c=c+1;
			I(c,:)=[n,m];
			% make the matrix hermitian
			CM(m,n,:)=conj(CM(n,m,:));
		end
		
	end
end

% removes the unused indeces
I=I(1:c,:);

% applies the normalization factor
NF=zeros(1,1,nf);
NF(1,1,:)=norm_factor;
CM=CM.*repmat(NF, [dims, dims, 1]);

% provides the output
varargout{1}=CM;
if nargout>1
	varargout{2}=I;
end