%
% Calculates the coupling matrix for elliptical core.
%
%   [K, I]=obj.calc_ellipticity(modes)
%   mi=obj.calc_ellipticity()
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
function varargout=calc_ellipticity(obj, varargin)


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
NA=obj.NA;
a=obj.radius;
lambda=obj.wavelength;
delta=a*obj.ellipticity.deltaRatio;

% define the functions
J=@(n,m,r) obj.J(sprintf('LP(%d,%d)', n, m), r);
dJ=@(n,m,r) obj.dJ(sprintf('LP(%d,%d)', n, m), r);
Q=@(n,m) obj.Q(sprintf('LP(%d,%d)', n, m));
beta=@(n,m) obj.propagationConst(sprintf('LP(%d,%d)', n, m));


% the perturbation matrix for the transverse components
% perturbation=[c.^(-2).*delta.*mu0.^(-1).*NA.^2.*cos(2.*phi),0,0;0,c.^(-2).* ...
%   delta.*mu0.^(-1).*NA.^2.*cos(2.*phi),0;0,0,0];

% the perturbation matrix for the longitudinal components
% perturbation=[0,0,0;0,0,0;0,0,c.^(-2).*delta.*mu0.^(-1).*NA.^2.*cos(2.*phi)];

% normalization factor; this multiplies the coupling matrix
% WARNING: the normalization factor given by Mathematica must be multiplied
%          by sqrt(-1)! This is needed to make the resulting coupling
%          consistent with the definition given in method 'couplingMatrix'.
%          The expression reported below is already corrected by the factor
%          sqrt(-1).
norm_factor=-a.*delta.*lambda.^(-2).*NA.^2.*pi.^2;


%--------------------------------------------------------------------------
% transverse components
%--------------------------------------------------------------------------

% define the coupling matrix as a cell array
% only the elements of the upper 'diagonal' are specified
% only non-zero elements are specified
Kt=cell(16,16);

% the non-zero elements
Kt{1,7}=(-1).*2.^(1/2).*J(0,1,a).*J(2,1,a).*(beta(0,1).*Q(0,1)).^(-1/2).*( ...
  beta(2,1).*Q(2,1)).^(-1/2);
Kt{2,8}=(-1).*2.^(1/2).*J(0,1,a).*J(2,1,a).*(beta(0,1).*Q(0,1)).^(-1/2).*( ...
  beta(2,1).*Q(2,1)).^(-1/2);
Kt{3,3}=(-1).*beta(1,1).^(-1).*J(1,1,a).^2.*Q(1,1).^(-1);
Kt{3,13}=(-1).*J(1,1,a).*J(3,1,a).*(beta(1,1).*Q(1,1)).^(-1/2).*(beta(3,1) ...
  .*Q(3,1)).^(-1/2);
Kt{4,4}=(-1).*beta(1,1).^(-1).*J(1,1,a).^2.*Q(1,1).^(-1);
Kt{4,14}=(-1).*J(1,1,a).*J(3,1,a).*(beta(1,1).*Q(1,1)).^(-1/2).*(beta(3,1) ...
  .*Q(3,1)).^(-1/2);
Kt{5,5}=beta(1,1).^(-1).*J(1,1,a).^2.*Q(1,1).^(-1);
Kt{5,15}=(-1).*J(1,1,a).*J(3,1,a).*(beta(1,1).*Q(1,1)).^(-1/2).*(beta(3,1) ...
  .*Q(3,1)).^(-1/2);
Kt{6,6}=beta(1,1).^(-1).*J(1,1,a).^2.*Q(1,1).^(-1);
Kt{6,16}=(-1).*J(1,1,a).*J(3,1,a).*(beta(1,1).*Q(1,1)).^(-1/2).*(beta(3,1) ...
  .*Q(3,1)).^(-1/2);
Kt{7,11}=(-1).*2.^(1/2).*J(0,2,a).*J(2,1,a).*(beta(0,2).*Q(0,2)).^(-1/2).*( ...
  beta(2,1).*Q(2,1)).^(-1/2);
Kt{8,12}=(-1).*2.^(1/2).*J(0,2,a).*J(2,1,a).*(beta(0,2).*Q(0,2)).^(-1/2).*( ...
  beta(2,1).*Q(2,1)).^(-1/2);


%--------------------------------------------------------------------------
% longitudinal components
%--------------------------------------------------------------------------

% define the coupling matrix as a cell array
% only the elements of the upper 'diagonal' are specified
% only non-zero elements are specified
Kl=cell(16,16);

% the non-zero elements
Kl{1,1}=(-1/2).*beta(0,1).^(-3).*dJ(0,1,a).^2.*Q(0,1).^(-1);
Kl{1,7}=(-1).*2.^(-1/2).*dJ(0,1,a).*dJ(2,1,a).*Q(0,1).*(beta(0,1).*Q(0,1)) ...
  .^(-3/2).*Q(2,1).*(beta(2,1).*Q(2,1)).^(-3/2);
Kl{1,10}=(-1).*2.^(1/2).*a.^(-1).*dJ(0,1,a).*J(2,1,a).*Q(0,1).*(beta(0,1).* ...
  Q(0,1)).^(-3/2).*Q(2,1).*(beta(2,1).*Q(2,1)).^(-3/2);
Kl{1,11}=(-1/2).*dJ(0,1,a).*dJ(0,2,a).*Q(0,1).*(beta(0,1).*Q(0,1)).^(-3/2) ...
  .*Q(0,2).*(beta(0,2).*Q(0,2)).^(-3/2);
Kl{2,2}=(1/2).*beta(0,1).^(-3).*dJ(0,1,a).^2.*Q(0,1).^(-1);
Kl{2,8}=(-1).*2.^(-1/2).*dJ(0,1,a).*dJ(2,1,a).*Q(0,1).*(beta(0,1).*Q(0,1)) ...
  .^(-3/2).*Q(2,1).*(beta(2,1).*Q(2,1)).^(-3/2);
Kl{2,9}=2.^(1/2).*a.^(-1).*dJ(0,1,a).*J(2,1,a).*Q(0,1).*(beta(0,1).*Q(0,1) ...
  ).^(-3/2).*Q(2,1).*(beta(2,1).*Q(2,1)).^(-3/2);
Kl{2,12}=(1/2).*dJ(0,1,a).*dJ(0,2,a).*Q(0,1).*(beta(0,1).*Q(0,1)).^(-3/2).* ...
  Q(0,2).*(beta(0,2).*Q(0,2)).^(-3/2);
Kl{3,3}=(-1).*beta(1,1).^(-3).*(dJ(1,1,a).^2+(-1).*a.^(-2).*J(1,1,a).^2).* ...
  Q(1,1).^(-1);
Kl{3,13}=(-1/4).*a.^(-2).*(3.*a.*dJ(1,1,a).*(a.*dJ(3,1,a)+J(3,1,a))+J(1,1, ...
  a).*(a.*dJ(3,1,a)+9.*J(3,1,a))).*Q(1,1).*(beta(1,1).*Q(1,1)).^( ...
  -3/2).*Q(3,1).*(beta(3,1).*Q(3,1)).^(-3/2);
Kl{3,16}=(-1/4).*a.^(-2).*(3.*J(1,1,a).*(a.*dJ(3,1,a)+J(3,1,a))+a.*dJ(1,1, ...
  a).*(a.*dJ(3,1,a)+9.*J(3,1,a))).*Q(1,1).*(beta(1,1).*Q(1,1)).^( ...
  -3/2).*Q(3,1).*(beta(3,1).*Q(3,1)).^(-3/2);
Kl{4,14}=(1/4).*a.^(-2).*((-1).*a.*dJ(1,1,a)+J(1,1,a)).*(a.*dJ(3,1,a)+(-3) ...
  .*J(3,1,a)).*Q(1,1).*(beta(1,1).*Q(1,1)).^(-3/2).*Q(3,1).*(beta(3, ...
  1).*Q(3,1)).^(-3/2);
Kl{4,15}=(1/4).*a.^(-2).*((-1).*a.*dJ(1,1,a)+J(1,1,a)).*(a.*dJ(3,1,a)+(-3) ...
  .*J(3,1,a)).*Q(1,1).*(beta(1,1).*Q(1,1)).^(-3/2).*Q(3,1).*(beta(3, ...
  1).*Q(3,1)).^(-3/2);
Kl{5,14}=(1/4).*a.^(-2).*((-1).*a.*dJ(1,1,a)+J(1,1,a)).*(a.*dJ(3,1,a)+(-3) ...
  .*J(3,1,a)).*Q(1,1).*(beta(1,1).*Q(1,1)).^(-3/2).*Q(3,1).*(beta(3, ...
  1).*Q(3,1)).^(-3/2);
Kl{5,15}=(1/4).*a.^(-2).*((-1).*a.*dJ(1,1,a)+J(1,1,a)).*(a.*dJ(3,1,a)+(-3) ...
  .*J(3,1,a)).*Q(1,1).*(beta(1,1).*Q(1,1)).^(-3/2).*Q(3,1).*(beta(3, ...
  1).*Q(3,1)).^(-3/2);
Kl{6,6}=(-1).*beta(1,1).^(-3).*((-1).*dJ(1,1,a).^2+a.^(-2).*J(1,1,a).^2).* ...
  Q(1,1).^(-1);
Kl{6,13}=(-1/4).*a.^(-2).*(3.*J(1,1,a).*(a.*dJ(3,1,a)+J(3,1,a))+a.*dJ(1,1, ...
  a).*(a.*dJ(3,1,a)+9.*J(3,1,a))).*Q(1,1).*(beta(1,1).*Q(1,1)).^( ...
  -3/2).*Q(3,1).*(beta(3,1).*Q(3,1)).^(-3/2);
Kl{6,16}=(-1/4).*a.^(-2).*(3.*a.*dJ(1,1,a).*(a.*dJ(3,1,a)+J(3,1,a))+J(1,1, ...
  a).*(a.*dJ(3,1,a)+9.*J(3,1,a))).*Q(1,1).*(beta(1,1).*Q(1,1)).^( ...
  -3/2).*Q(3,1).*(beta(3,1).*Q(3,1)).^(-3/2);
Kl{7,7}=(1/4).*a.^(-2).*beta(2,1).^(-3).*((-3).*a.*dJ(2,1,a)+2.*J(2,1,a)) ...
  .*(a.*dJ(2,1,a)+2.*J(2,1,a)).*Q(2,1).^(-1);
Kl{7,10}=(-1/4).*a.^(-2).*beta(2,1).^(-3).*(a.*dJ(2,1,a)+2.*J(2,1,a)).^2.* ...
  Q(2,1).^(-1);
Kl{7,11}=(-1).*2.^(-1/2).*dJ(0,2,a).*dJ(2,1,a).*Q(0,2).*(beta(0,2).*Q(0,2)) ...
  .^(-3/2).*Q(2,1).*(beta(2,1).*Q(2,1)).^(-3/2);
Kl{8,8}=(1/4).*a.^(-2).*beta(2,1).^(-3).*(3.*a.*dJ(2,1,a)+(-2).*J(2,1,a)) ...
  .*(a.*dJ(2,1,a)+2.*J(2,1,a)).*Q(2,1).^(-1);
Kl{8,9}=(-1/4).*a.^(-2).*beta(2,1).^(-3).*(a.*dJ(2,1,a)+2.*J(2,1,a)).^2.* ...
  Q(2,1).^(-1);
Kl{8,12}=(-1).*2.^(-1/2).*dJ(0,2,a).*dJ(2,1,a).*Q(0,2).*(beta(0,2).*Q(0,2)) ...
  .^(-3/2).*Q(2,1).*(beta(2,1).*Q(2,1)).^(-3/2);
Kl{9,9}=(-1/4).*a.^(-2).*beta(2,1).^(-3).*(a.*dJ(2,1,a)+(-6).*J(2,1,a)).*( ...
  a.*dJ(2,1,a)+2.*J(2,1,a)).*Q(2,1).^(-1);
Kl{9,12}=2.^(1/2).*a.^(-1).*dJ(0,2,a).*J(2,1,a).*Q(0,2).*(beta(0,2).*Q(0,2) ...
  ).^(-3/2).*Q(2,1).*(beta(2,1).*Q(2,1)).^(-3/2);
Kl{10,10}=(1/4).*a.^(-2).*beta(2,1).^(-3).*(a.*dJ(2,1,a)+(-6).*J(2,1,a)).*( ...
  a.*dJ(2,1,a)+2.*J(2,1,a)).*Q(2,1).^(-1);
Kl{10,11}=(-1).*2.^(1/2).*a.^(-1).*dJ(0,2,a).*J(2,1,a).*Q(0,2).*(beta(0,2).* ...
  Q(0,2)).^(-3/2).*Q(2,1).*(beta(2,1).*Q(2,1)).^(-3/2);
Kl{11,11}=(-1/2).*beta(0,2).^(-3).*dJ(0,2,a).^2.*Q(0,2).^(-1);
Kl{12,12}=(1/2).*beta(0,2).^(-3).*dJ(0,2,a).^2.*Q(0,2).^(-1);
Kl{13,13}=(-1/2).*beta(3,1).^(-3).*(dJ(3,1,a).^2+(-9).*a.^(-2).*J(3,1,a).^2) ...
  .*Q(3,1).^(-1);
Kl{14,14}=(1/2).*a.^(-2).*beta(3,1).^(-3).*(a.^2.*dJ(3,1,a).^2+(-9).*J(3,1, ...
  a).^2).*Q(3,1).^(-1);
Kl{15,15}=(-1/2).*beta(3,1).^(-3).*(dJ(3,1,a).^2+(-9).*a.^(-2).*J(3,1,a).^2) ...
  .*Q(3,1).^(-1);
Kl{16,16}=(1/2).*a.^(-2).*beta(3,1).^(-3).*(a.^2.*dJ(3,1,a).^2+(-9).*J(3,1, ...
  a).^2).*Q(3,1).^(-1);



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
		
		% calculate the transverse components
		if any(strcmp(obj.ellipticity.components, {'all', 'transverse'}))
			if ~isempty(Kt{indx(n),indx(m)})
				% there is something to calculate
				CM(n,m,:)=Kt{indx(n),indx(m)};
				nonzero=true;
			elseif ~isempty(Kt{indx(m),indx(n)})
				% we have to calculate the transpose conjugate
				CM(n,m,:)=conj(Kt{indx(m),indx(n)}); 
				nonzero=true;
			end
		end
		
		% calculate the longitudinal components
		if any(strcmp(obj.ellipticity.components, {'all', 'longitudinal'}))	
			if ~isempty(Kl{indx(n),indx(m)})
				% there is something to calculate
				CM(n,m,:)=squeeze(CM(n,m,:))' + Kl{indx(n),indx(m)};
				nonzero=true;
			elseif ~isempty(Kl{indx(m),indx(n)})
				% we have to calculate the transpose
				CM(n,m,:)=squeeze(CM(n,m,:))' + conj(Kl{indx(m),indx(n)}); 
				nonzero=true;
			end
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

% removed the unused indeces
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