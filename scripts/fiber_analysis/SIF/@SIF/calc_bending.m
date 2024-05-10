%
% Calculates the coupling matrix due to bending.
%
%   [K, I]=obj.calc_bending(modes)
%   mi=obj.calc_bending()
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
function varargout=calc_bending(obj, varargin)


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
nav=(obj.nco+obj.ncl)/2;
Rcl=obj.radiusCL;
if isempty(Rcl)
    error('SIF:wrongvalue', 'The cladding radius has not been defined.')
end
lambda=obj.wavelength;
kappa=1/obj.bending.radius;
q1=0.206;
q2=0.032;

% define the functions
J=@(n,m,r) obj.J(sprintf('LP(%d,%d)', n, m), r);
dJ=@(n,m,r) obj.dJ(sprintf('LP(%d,%d)', n, m), r);
Q=@(n,m) obj.Q(sprintf('LP(%d,%d)', n, m));
beta=@(n,m) obj.propagationConst(sprintf('LP(%d,%d)', n, m));

% the perturbation matrix due to:
% longitudinal strain on transverse components:
% perturbation=[(-1).*c.^(-2).*kappa.*mu0.^(-1).*nav.^4.*q1.*r.*cos(phi),0,0;
%               0,(-1).*c.^(-2).*kappa.*mu0.^(-1).*nav.^4.*q1.*r.*cos(phi),0;
%               0,0,0]; 
%
% transverse strain on transverse components
% perturbation=[(1/2).*kappa.^2.*nav.^4.*q2.*Rcl.^2,0,0;
%               0,(1/2).*kappa.^2.nav.^4.*q1.*Rcl.^2,0; 0,0,0];
%
% transverse strain on longitudinal components
% perturbation=[0,0,0; 0,0,0;
%               0,0,(1/2).*c.^(-2).*kappa.^2.*mu0.^(-1).*nav.^4.*q1.*Rcl.^2];
%
% longitudinal strain on longitudinal components
% perturbation=[0,0,0; 0,0,0; 
%               0,0,(-1).*c.^(-2).*kappa.*mu0.^(-1).*nav.^4.*q2.*r.*cos(phi)];

% normalization factor; this multiplies the coupling matrix
% WARNING: the normalization factor given by Mathematica must be multiplied
%          by sqrt(-1)! This is needed to make the resulting coupling
%          consistent with the definition given in method 'couplingMatrix'.
%          The expression reported below is already corrected by the factor
%          sqrt(-1).
norm_factor=lambda.^(-2).*nav.^4;


%--------------------------------------------------------------------------
% transverse components
%--------------------------------------------------------------------------

% define the coupling matrix as a cell array of handle function
% only the elements of the upper 'triangle' are specified
% only non-zero elements are specified
Kzt=cell(16,16);

% the non-zero elements
Kzt{1,3}=@(r) 2.^(1/2).*kappa.*pi.^2.*q1.*r.^2.*J(0,1,r).*J(1,1,r).*(beta(0,1).* ...
  Q(0,1)).^(-1/2).*(beta(1,1).*Q(1,1)).^(-1/2);
Kzt{2,4}=@(r) 2.^(1/2).*kappa.*pi.^2.*q1.*r.^2.*J(0,1,r).*J(1,1,r).*(beta(0,1).* ...
  Q(0,1)).^(-1/2).*(beta(1,1).*Q(1,1)).^(-1/2);
Kzt{3,7}=@(r) kappa.*pi.^2.*q1.*r.^2.*J(1,1,r).*J(2,1,r).*(beta(1,1).*Q(1,1)).^( ...
  -1/2).*(beta(2,1).*Q(2,1)).^(-1/2);
Kzt{3,11}=@(r) 2.^(1/2).*kappa.*pi.^2.*q1.*r.^2.*J(0,2,r).*J(1,1,r).*(beta(0,2).* ...
  Q(0,2)).^(-1/2).*(beta(1,1).*Q(1,1)).^(-1/2);
Kzt{4,8}=@(r) kappa.*pi.^2.*q1.*r.^2.*J(1,1,r).*J(2,1,r).*(beta(1,1).*Q(1,1)).^( ...
  -1/2).*(beta(2,1).*Q(2,1)).^(-1/2);
Kzt{4,12}=@(r) 2.^(1/2).*kappa.*pi.^2.*q1.*r.^2.*J(0,2,r).*J(1,1,r).*(beta(0,2).* ...
  Q(0,2)).^(-1/2).*(beta(1,1).*Q(1,1)).^(-1/2);
Kzt{5,9}=@(r) kappa.*pi.^2.*q1.*r.^2.*J(1,1,r).*J(2,1,r).*(beta(1,1).*Q(1,1)).^( ...
  -1/2).*(beta(2,1).*Q(2,1)).^(-1/2);
Kzt{6,10}=@(r) kappa.*pi.^2.*q1.*r.^2.*J(1,1,r).*J(2,1,r).*(beta(1,1).*Q(1,1)).^( ...
  -1/2).*(beta(2,1).*Q(2,1)).^(-1/2);
Kzt{7,13}=@(r) kappa.*pi.^2.*q1.*r.^2.*J(2,1,r).*J(3,1,r).*(beta(2,1).*Q(2,1)).^( ...
  -1/2).*(beta(3,1).*Q(3,1)).^(-1/2);
Kzt{8,14}=@(r) kappa.*pi.^2.*q1.*r.^2.*J(2,1,r).*J(3,1,r).*(beta(2,1).*Q(2,1)).^( ...
  -1/2).*(beta(3,1).*Q(3,1)).^(-1/2);
Kzt{9,15}=@(r) kappa.*pi.^2.*q1.*r.^2.*J(2,1,r).*J(3,1,r).*(beta(2,1).*Q(2,1)).^( ...
  -1/2).*(beta(3,1).*Q(3,1)).^(-1/2);
Kzt{10,16}=@(r) kappa.*pi.^2.*q1.*r.^2.*J(2,1,r).*J(3,1,r).*(beta(2,1).*Q(2,1)).^( ...
  -1/2).*(beta(3,1).*Q(3,1)).^(-1/2);

% define the coupling matrix as a cell array
% only the elements of the upper 'diagonal' are specified
% only non-zero elements are specified
Ktt=cell(16,16);

% the non-zero elements (off-diagonal elements have been removed by hand,
% because they are 0 by orthogonality)
Ktt{1,1}=(-1).*kappa.^2.*pi.^2.*q2.*Rcl.^2.*beta(0,1).^(-1);
Ktt{2,2}=(-1).*kappa.^2.*pi.^2.*q1.*Rcl.^2.*beta(0,1).^(-1);
Ktt{3,3}=(-1).*kappa.^2.*pi.^2.*q2.*Rcl.^2.*beta(1,1).^(-1);
Ktt{4,4}=(-1).*kappa.^2.*pi.^2.*q1.*Rcl.^2.*beta(1,1).^(-1);
Ktt{5,5}=(-1).*kappa.^2.*pi.^2.*q2.*Rcl.^2.*beta(1,1).^(-1);
Ktt{6,6}=(-1).*kappa.^2.*pi.^2.*q1.*Rcl.^2.*beta(1,1).^(-1);
Ktt{7,7}=(-1).*kappa.^2.*pi.^2.*q2.*Rcl.^2.*beta(2,1).^(-1);
Ktt{8,8}=(-1).*kappa.^2.*pi.^2.*q1.*Rcl.^2.*beta(2,1).^(-1);
Ktt{9,9}=(-1).*kappa.^2.*pi.^2.*q2.*Rcl.^2.*beta(2,1).^(-1);
Ktt{10,10}=(-1).*kappa.^2.*pi.^2.*q1.*Rcl.^2.*beta(2,1).^(-1);
Ktt{11,11}=(-1).*kappa.^2.*pi.^2.*q2.*Rcl.^2.*beta(0,2).^(-1);
Ktt{12,12}=(-1).*kappa.^2.*pi.^2.*q1.*Rcl.^2.*beta(0,2).^(-1);
Ktt{13,13}=(-1).*kappa.^2.*pi.^2.*q2.*Rcl.^2.*beta(3,1).^(-1);
Ktt{14,14}=(-1).*kappa.^2.*pi.^2.*q1.*Rcl.^2.*beta(3,1).^(-1);
Ktt{15,15}=(-1).*kappa.^2.*pi.^2.*q2.*Rcl.^2.*beta(3,1).^(-1);
Ktt{16,16}=(-1).*kappa.^2.*pi.^2.*q1.*Rcl.^2.*beta(3,1).^(-1);


%--------------------------------------------------------------------------
% longitudinal components
%--------------------------------------------------------------------------

% define the coupling matrix as a cell array
% only the elements of the upper 'diagonal' are specified
% only non-zero elements are specified
Ktz=cell(16,16);

% the non-zero elements
Ktz{1,1}=@(r) (-1/2).*kappa.^2.*pi.^2.*q1.*r.*Rcl.^2.*beta(0,1).^(-3).*dJ(0,1,r) ...
  .^2.*Q(0,1).^(-1);
Ktz{1,7}=@(r) (-1/2).*2.^(-1/2).*kappa.^2.*pi.^2.*q1.*Rcl.^2.*dJ(0,1,r).*(r.*dJ( ...
  2,1,r)+2.*J(2,1,r)).*Q(0,1).*(beta(0,1).*Q(0,1)).^(-3/2).*Q(2,1).* ...
  (beta(2,1).*Q(2,1)).^(-3/2);
Ktz{1,10}=@(r) (-1/2).*2.^(-1/2).*kappa.^2.*pi.^2.*q1.*Rcl.^2.*dJ(0,1,r).*(r.*dJ( ...
  2,1,r)+2.*J(2,1,r)).*Q(0,1).*(beta(0,1).*Q(0,1)).^(-3/2).*Q(2,1).* ...
  (beta(2,1).*Q(2,1)).^(-3/2);
Ktz{1,11}=@(r) (-1/2).*kappa.^2.*pi.^2.*q1.*r.*Rcl.^2.*dJ(0,1,r).*dJ(0,2,r).*Q(0, ...
  1).*(beta(0,1).*Q(0,1)).^(-3/2).*Q(0,2).*(beta(0,2).*Q(0,2)).^( ...
  -3/2);
Ktz{2,2}=@(r) (-1/2).*kappa.^2.*pi.^2.*q1.*r.*Rcl.^2.*beta(0,1).^(-3).*dJ(0,1,r) ...
  .^2.*Q(0,1).^(-1);
Ktz{2,8}=@(r) (1/2).*2.^(-1/2).*kappa.^2.*pi.^2.*q1.*Rcl.^2.*dJ(0,1,r).*(r.*dJ( ...
  2,1,r)+2.*J(2,1,r)).*Q(0,1).*(beta(0,1).*Q(0,1)).^(-3/2).*Q(2,1).* ...
  (beta(2,1).*Q(2,1)).^(-3/2);
Ktz{2,9}=@(r) (-1/2).*2.^(-1/2).*kappa.^2.*pi.^2.*q1.*Rcl.^2.*dJ(0,1,r).*(r.*dJ( ...
  2,1,r)+2.*J(2,1,r)).*Q(0,1).*(beta(0,1).*Q(0,1)).^(-3/2).*Q(2,1).* ...
  (beta(2,1).*Q(2,1)).^(-3/2);
Ktz{2,12}=@(r) (-1/2).*kappa.^2.*pi.^2.*q1.*r.*Rcl.^2.*dJ(0,1,r).*dJ(0,2,r).*Q(0, ...
  1).*(beta(0,1).*Q(0,1)).^(-3/2).*Q(0,2).*(beta(0,2).*Q(0,2)).^( ...
  -3/2);
Ktz{3,3}=@(r) (-1/4).*kappa.^2.*pi.^2.*q1.*r.^(-1).*Rcl.^2.*beta(1,1).^(-3).*( ...
  3.*r.^2.*dJ(1,1,r).^2+2.*r.*dJ(1,1,r).*J(1,1,r)+3.*J(1,1,r).^2).* ...
  Q(1,1).^(-1);
Ktz{3,6}=@(r) (-1/4).*kappa.^2.*pi.^2.*q1.*r.^(-1).*Rcl.^2.*beta(1,1).^(-3).*( ...
  r.^2.*dJ(1,1,r).^2+6.*r.*dJ(1,1,r).*J(1,1,r)+J(1,1,r).^2).*Q(1,1) ...
  .^(-1);
Ktz{3,13}=@(r) (1/4).*kappa.^2.*pi.^2.*q1.*r.^(-1).*Rcl.^2.*((-1).*r.*dJ(1,1,r)+ ...
  J(1,1,r)).*(r.*dJ(3,1,r)+3.*J(3,1,r)).*Q(1,1).*(beta(1,1).*Q(1,1)) ...
  .^(-3/2).*Q(3,1).*(beta(3,1).*Q(3,1)).^(-3/2);
Ktz{3,16}=@(r) (1/4).*kappa.^2.*pi.^2.*q1.*r.^(-1).*Rcl.^2.*((-1).*r.*dJ(1,1,r)+ ...
  J(1,1,r)).*(r.*dJ(3,1,r)+3.*J(3,1,r)).*Q(1,1).*(beta(1,1).*Q(1,1)) ...
  .^(-3/2).*Q(3,1).*(beta(3,1).*Q(3,1)).^(-3/2);
Ktz{4,4}=@(r) (-1/4).*kappa.^2.*pi.^2.*q1.*r.^(-1).*Rcl.^2.*beta(1,1).^(-3).*(( ...
  -1).*r.*dJ(1,1,r)+J(1,1,r)).^2.*Q(1,1).^(-1);
Ktz{4,5}=@(r) (-1/4).*kappa.^2.*pi.^2.*q1.*r.^(-1).*Rcl.^2.*beta(1,1).^(-3).*(( ...
  -1).*r.*dJ(1,1,r)+J(1,1,r)).^2.*Q(1,1).^(-1);
Ktz{4,14}=@(r) (1/4).*kappa.^2.*pi.^2.*q1.*r.^(-1).*Rcl.^2.*(r.*dJ(1,1,r)+(-1).* ...
  J(1,1,r)).*(r.*dJ(3,1,r)+3.*J(3,1,r)).*Q(1,1).*(beta(1,1).*Q(1,1)) ...
  .^(-3/2).*Q(3,1).*(beta(3,1).*Q(3,1)).^(-3/2);
Ktz{4,15}=@(r) (1/4).*kappa.^2.*pi.^2.*q1.*r.^(-1).*Rcl.^2.*((-1).*r.*dJ(1,1,r)+ ...
  J(1,1,r)).*(r.*dJ(3,1,r)+3.*J(3,1,r)).*Q(1,1).*(beta(1,1).*Q(1,1)) ...
  .^(-3/2).*Q(3,1).*(beta(3,1).*Q(3,1)).^(-3/2);
Ktz{5,5}=@(r) (-1/4).*kappa.^2.*pi.^2.*q1.*r.^(-1).*Rcl.^2.*beta(1,1).^(-3).*(( ...
  -1).*r.*dJ(1,1,r)+J(1,1,r)).^2.*Q(1,1).^(-1);
Ktz{5,14}=@(r) (1/4).*kappa.^2.*pi.^2.*q1.*r.^(-1).*Rcl.^2.*(r.*dJ(1,1,r)+(-1).* ...
  J(1,1,r)).*(r.*dJ(3,1,r)+3.*J(3,1,r)).*Q(1,1).*(beta(1,1).*Q(1,1)) ...
  .^(-3/2).*Q(3,1).*(beta(3,1).*Q(3,1)).^(-3/2);
Ktz{5,15}=@(r) (1/4).*kappa.^2.*pi.^2.*q1.*r.^(-1).*Rcl.^2.*((-1).*r.*dJ(1,1,r)+ ...
  J(1,1,r)).*(r.*dJ(3,1,r)+3.*J(3,1,r)).*Q(1,1).*(beta(1,1).*Q(1,1)) ...
  .^(-3/2).*Q(3,1).*(beta(3,1).*Q(3,1)).^(-3/2);
Ktz{6,6}=@(r) (-1/4).*kappa.^2.*pi.^2.*q1.*r.^(-1).*Rcl.^2.*beta(1,1).^(-3).*( ...
  3.*r.^2.*dJ(1,1,r).^2+2.*r.*dJ(1,1,r).*J(1,1,r)+3.*J(1,1,r).^2).* ...
  Q(1,1).^(-1);
Ktz{6,13}=@(r) (1/4).*kappa.^2.*pi.^2.*q1.*r.^(-1).*Rcl.^2.*(r.*dJ(1,1,r)+(-1).* ...
  J(1,1,r)).*(r.*dJ(3,1,r)+3.*J(3,1,r)).*Q(1,1).*(beta(1,1).*Q(1,1)) ...
  .^(-3/2).*Q(3,1).*(beta(3,1).*Q(3,1)).^(-3/2);
Ktz{6,16}=@(r) (1/4).*kappa.^2.*pi.^2.*q1.*r.^(-1).*Rcl.^2.*(r.*dJ(1,1,r)+(-1).* ...
  J(1,1,r)).*(r.*dJ(3,1,r)+3.*J(3,1,r)).*Q(1,1).*(beta(1,1).*Q(1,1)) ...
  .^(-3/2).*Q(3,1).*(beta(3,1).*Q(3,1)).^(-3/2);
Ktz{7,7}=@(r) (-1/2).*kappa.^2.*pi.^2.*q1.*r.^(-1).*Rcl.^2.*beta(2,1).^(-3).*( ...
  r.^2.*dJ(2,1,r).^2+4.*J(2,1,r).^2).*Q(2,1).^(-1);
Ktz{7,10}=@(r) (-2).*kappa.^2.*pi.^2.*q1.*Rcl.^2.*beta(2,1).^(-3).*dJ(2,1,r).*J( ...
  2,1,r).*Q(2,1).^(-1);
Ktz{7,11}=@(r) (-1/2).*2.^(-1/2).*kappa.^2.*pi.^2.*q1.*Rcl.^2.*dJ(0,2,r).*(r.*dJ( ...
  2,1,r)+2.*J(2,1,r)).*Q(0,2).*(beta(0,2).*Q(0,2)).^(-3/2).*Q(2,1).* ...
  (beta(2,1).*Q(2,1)).^(-3/2);
Ktz{8,8}=@(r) (-1/2).*kappa.^2.*pi.^2.*q1.*r.^(-1).*Rcl.^2.*beta(2,1).^(-3).*( ...
  r.^2.*dJ(2,1,r).^2+4.*J(2,1,r).^2).*Q(2,1).^(-1);
Ktz{8,9}=@(r) 2.*kappa.^2.*pi.^2.*q1.*Rcl.^2.*beta(2,1).^(-3).*dJ(2,1,r).*J(2,1, ...
  r).*Q(2,1).^(-1);
Ktz{8,12}=@(r) (1/2).*2.^(-1/2).*kappa.^2.*pi.^2.*q1.*Rcl.^2.*dJ(0,2,r).*(r.*dJ( ...
  2,1,r)+2.*J(2,1,r)).*Q(0,2).*(beta(0,2).*Q(0,2)).^(-3/2).*Q(2,1).* ...
  (beta(2,1).*Q(2,1)).^(-3/2);
Ktz{9,9}=@(r) (-1/2).*kappa.^2.*pi.^2.*q1.*r.^(-1).*Rcl.^2.*beta(2,1).^(-3).*( ...
  r.^2.*dJ(2,1,r).^2+4.*J(2,1,r).^2).*Q(2,1).^(-1);
Ktz{9,12}=@(r) (-1/2).*2.^(-1/2).*kappa.^2.*pi.^2.*q1.*Rcl.^2.*dJ(0,2,r).*(r.*dJ( ...
  2,1,r)+2.*J(2,1,r)).*Q(0,2).*(beta(0,2).*Q(0,2)).^(-3/2).*Q(2,1).* ...
  (beta(2,1).*Q(2,1)).^(-3/2);
Ktz{10,10}=@(r) (-1/2).*kappa.^2.*pi.^2.*q1.*r.*Rcl.^2.*beta(2,1).^(-3).*(dJ(2,1, ...
  r).^2+4.*r.^(-2).*J(2,1,r).^2).*Q(2,1).^(-1);
Ktz{10,11}=@(r) (-1/2).*2.^(-1/2).*kappa.^2.*pi.^2.*q1.*Rcl.^2.*dJ(0,2,r).*(r.*dJ( ...
  2,1,r)+2.*J(2,1,r)).*Q(0,2).*(beta(0,2).*Q(0,2)).^(-3/2).*Q(2,1).* ...
  (beta(2,1).*Q(2,1)).^(-3/2);
Ktz{11,11}=@(r) (-1/2).*kappa.^2.*pi.^2.*q1.*r.*Rcl.^2.*beta(0,2).^(-3).*dJ(0,2,r) ...
  .^2.*Q(0,2).^(-1);
Ktz{12,12}=@(r) (-1/2).*kappa.^2.*pi.^2.*q1.*r.*Rcl.^2.*beta(0,2).^(-3).*dJ(0,2,r) ...
  .^2.*Q(0,2).^(-1);
Ktz{13,13}=@(r) (-1/2).*kappa.^2.*pi.^2.*q1.*r.*Rcl.^2.*beta(3,1).^(-3).*(dJ(3,1, ...
  r).^2+9.*r.^(-2).*J(3,1,r).^2).*Q(3,1).^(-1);
Ktz{13,16}=@(r) (-3).*kappa.^2.*pi.^2.*q1.*Rcl.^2.*beta(3,1).^(-3).*dJ(3,1,r).*J( ...
  3,1,r).*Q(3,1).^(-1);
Ktz{14,14}=@(r) (-1/2).*kappa.^2.*pi.^2.*q1.*r.*Rcl.^2.*beta(3,1).^(-3).*(dJ(3,1, ...
  r).^2+9.*r.^(-2).*J(3,1,r).^2).*Q(3,1).^(-1);
Ktz{14,15}=@(r) 3.*kappa.^2.*pi.^2.*q1.*Rcl.^2.*beta(3,1).^(-3).*dJ(3,1,r).*J(3,1, ...
  r).*Q(3,1).^(-1);
Ktz{15,15}=@(r) (-1/2).*kappa.^2.*pi.^2.*q1.*r.*Rcl.^2.*beta(3,1).^(-3).*(dJ(3,1, ...
  r).^2+9.*r.^(-2).*J(3,1,r).^2).*Q(3,1).^(-1);
Ktz{16,16}=@(r) (-1/2).*kappa.^2.*pi.^2.*q1.*r.*Rcl.^2.*beta(3,1).^(-3).*(dJ(3,1, ...
  r).^2+9.*r.^(-2).*J(3,1,r).^2).*Q(3,1).^(-1);

% define the coupling matrix as a cell array
% only the elements of the upper 'diagonal' are specified
% only non-zero elements are specified
Kzz=cell(16,16);

% the non-zero elements
Kzz{1,3}=@(r) (1/2).*2.^(-1/2).*kappa.*pi.^2.*q2.*r.*dJ(0,1,r).*(3.*r.*dJ(1,1,r) ...
  +J(1,1,r)).*Q(0,1).*(beta(0,1).*Q(0,1)).^(-3/2).*Q(1,1).*(beta(1, ...
  1).*Q(1,1)).^(-3/2);
Kzz{1,6}=@(r) (1/2).*2.^(-1/2).*kappa.*pi.^2.*q2.*r.*dJ(0,1,r).*(r.*dJ(1,1,r)+ ...
  3.*J(1,1,r)).*Q(0,1).*(beta(0,1).*Q(0,1)).^(-3/2).*Q(1,1).*(beta( ...
  1,1).*Q(1,1)).^(-3/2);
Kzz{1,13}=@(r) (1/2).*2.^(-1/2).*kappa.*pi.^2.*q2.*r.*dJ(0,1,r).*(r.*dJ(3,1,r)+ ...
  3.*J(3,1,r)).*Q(0,1).*(beta(0,1).*Q(0,1)).^(-3/2).*Q(3,1).*(beta( ...
  3,1).*Q(3,1)).^(-3/2);
Kzz{1,16}=@(r) (1/2).*2.^(-1/2).*kappa.*pi.^2.*q2.*r.*dJ(0,1,r).*(r.*dJ(3,1,r)+ ...
  3.*J(3,1,r)).*Q(0,1).*(beta(0,1).*Q(0,1)).^(-3/2).*Q(3,1).*(beta( ...
  3,1).*Q(3,1)).^(-3/2);
Kzz{2,4}=@(r) (1/2).*2.^(-1/2).*kappa.*pi.^2.*q2.*r.*dJ(0,1,r).*(r.*dJ(1,1,r)+( ...
  -1).*J(1,1,r)).*Q(0,1).*(beta(0,1).*Q(0,1)).^(-3/2).*Q(1,1).*( ...
  beta(1,1).*Q(1,1)).^(-3/2);
Kzz{2,5}=@(r) (1/2).*2.^(-1/2).*kappa.*pi.^2.*q2.*r.*dJ(0,1,r).*(r.*dJ(1,1,r)+( ...
  -1).*J(1,1,r)).*Q(0,1).*(beta(0,1).*Q(0,1)).^(-3/2).*Q(1,1).*( ...
  beta(1,1).*Q(1,1)).^(-3/2);
Kzz{2,14}=@(r) (-1/2).*2.^(-1/2).*kappa.*pi.^2.*q2.*r.*dJ(0,1,r).*(r.*dJ(3,1,r)+ ...
  3.*J(3,1,r)).*Q(0,1).*(beta(0,1).*Q(0,1)).^(-3/2).*Q(3,1).*(beta( ...
  3,1).*Q(3,1)).^(-3/2);
Kzz{2,15}=@(r) (1/2).*2.^(-1/2).*kappa.*pi.^2.*q2.*r.*dJ(0,1,r).*(r.*dJ(3,1,r)+ ...
  3.*J(3,1,r)).*Q(0,1).*(beta(0,1).*Q(0,1)).^(-3/2).*Q(3,1).*(beta( ...
  3,1).*Q(3,1)).^(-3/2);
Kzz{3,7}=@(r) kappa.*pi.^2.*q2.*(J(1,1,r).*J(2,1,r)+r.*dJ(1,1,r).*(r.*dJ(2,1,r)+ ...
  J(2,1,r))).*Q(1,1).*(beta(1,1).*Q(1,1)).^(-3/2).*Q(2,1).*(beta(2, ...
  1).*Q(2,1)).^(-3/2);
Kzz{3,10}=@(r) (1/2).*kappa.*pi.^2.*q2.*r.*(dJ(2,1,r).*(r.*dJ(1,1,r)+J(1,1,r))+ ...
  4.*dJ(1,1,r).*J(2,1,r)).*Q(1,1).*(beta(1,1).*Q(1,1)).^(-3/2).*Q(2, ...
  1).*(beta(2,1).*Q(2,1)).^(-3/2);
Kzz{3,11}=@(r) (1/2).*2.^(-1/2).*kappa.*pi.^2.*q2.*r.*dJ(0,2,r).*(3.*r.*dJ(1,1,r) ...
  +J(1,1,r)).*Q(0,2).*(beta(0,2).*Q(0,2)).^(-3/2).*Q(1,1).*(beta(1, ...
  1).*Q(1,1)).^(-3/2);
Kzz{4,8}=@(r) kappa.*pi.^2.*q2.*((-1).*r.*dJ(1,1,r)+J(1,1,r)).*J(2,1,r).*Q(1,1) ...
  .*(beta(1,1).*Q(1,1)).^(-3/2).*Q(2,1).*(beta(2,1).*Q(2,1)).^(-3/2) ...
  ;
Kzz{4,9}=@(r) (1/2).*kappa.*pi.^2.*q2.*r.*dJ(2,1,r).*(r.*dJ(1,1,r)+(-1).*J(1,1, ...
  r)).*Q(1,1).*(beta(1,1).*Q(1,1)).^(-3/2).*Q(2,1).*(beta(2,1).*Q(2, ...
  1)).^(-3/2);
Kzz{4,12}=@(r) (1/2).*2.^(-1/2).*kappa.*pi.^2.*q2.*r.*dJ(0,2,r).*(r.*dJ(1,1,r)+( ...
  -1).*J(1,1,r)).*Q(0,2).*(beta(0,2).*Q(0,2)).^(-3/2).*Q(1,1).*( ...
  beta(1,1).*Q(1,1)).^(-3/2);
Kzz{5,8}=@(r) kappa.*pi.^2.*q2.*((-1).*r.*dJ(1,1,r)+J(1,1,r)).*J(2,1,r).*Q(1,1) ...
  .*(beta(1,1).*Q(1,1)).^(-3/2).*Q(2,1).*(beta(2,1).*Q(2,1)).^(-3/2) ...
  ;
Kzz{5,9}=@(r) (1/2).*kappa.*pi.^2.*q2.*r.*dJ(2,1,r).*(r.*dJ(1,1,r)+(-1).*J(1,1, ...
  r)).*Q(1,1).*(beta(1,1).*Q(1,1)).^(-3/2).*Q(2,1).*(beta(2,1).*Q(2, ...
  1)).^(-3/2);
Kzz{5,12}=@(r) (1/2).*2.^(-1/2).*kappa.*pi.^2.*q2.*r.*dJ(0,2,r).*(r.*dJ(1,1,r)+( ...
  -1).*J(1,1,r)).*Q(0,2).*(beta(0,2).*Q(0,2)).^(-3/2).*Q(1,1).*( ...
  beta(1,1).*Q(1,1)).^(-3/2);
Kzz{6,7}=@(r) kappa.*pi.^2.*q2.*(r.*dJ(1,1,r).*J(2,1,r)+J(1,1,r).*(r.*dJ(2,1,r)+ ...
  J(2,1,r))).*Q(1,1).*(beta(1,1).*Q(1,1)).^(-3/2).*Q(2,1).*(beta(2, ...
  1).*Q(2,1)).^(-3/2);
Kzz{6,10}=@(r) (1/2).*kappa.*pi.^2.*q2.*(r.*dJ(2,1,r).*(r.*dJ(1,1,r)+J(1,1,r))+ ...
  4.*J(1,1,r).*J(2,1,r)).*Q(1,1).*(beta(1,1).*Q(1,1)).^(-3/2).*Q(2, ...
  1).*(beta(2,1).*Q(2,1)).^(-3/2);
Kzz{6,11}=@(r) (1/2).*2.^(-1/2).*kappa.*pi.^2.*q2.*r.*dJ(0,2,r).*(r.*dJ(1,1,r)+ ...
  3.*J(1,1,r)).*Q(0,2).*(beta(0,2).*Q(0,2)).^(-3/2).*Q(1,1).*(beta( ...
  1,1).*Q(1,1)).^(-3/2);
Kzz{7,13}=@(r) (1/4).*kappa.*pi.^2.*q2.*(3.*r.*dJ(2,1,r).*(r.*dJ(3,1,r)+J(3,1,r)) ...
  +J(2,1,r).*((-2).*r.*dJ(3,1,r)+6.*J(3,1,r))).*Q(2,1).*(beta(2,1).* ...
  Q(2,1)).^(-3/2).*Q(3,1).*(beta(3,1).*Q(3,1)).^(-3/2);
Kzz{7,16}=@(r) (1/4).*kappa.*pi.^2.*q2.*(J(2,1,r).*(2.*r.*dJ(3,1,r)+(-6).*J(3,1, ...
  r))+r.*dJ(2,1,r).*(r.*dJ(3,1,r)+9.*J(3,1,r))).*Q(2,1).*(beta(2,1) ...
  .*Q(2,1)).^(-3/2).*Q(3,1).*(beta(3,1).*Q(3,1)).^(-3/2);
Kzz{8,14}=@(r) (1/4).*kappa.*pi.^2.*q2.*(r.*dJ(2,1,r).*(r.*dJ(3,1,r)+(-3).*J(3,1, ...
  r))+2.*J(2,1,r).*(r.*dJ(3,1,r)+9.*J(3,1,r))).*Q(2,1).*(beta(2,1).* ...
  Q(2,1)).^(-3/2).*Q(3,1).*(beta(3,1).*Q(3,1)).^(-3/2);
Kzz{8,15}=@(r) (-1/4).*kappa.*pi.^2.*q2.*(6.*J(2,1,r).*(r.*dJ(3,1,r)+J(3,1,r))+ ...
  r.*dJ(2,1,r).*((-1).*r.*dJ(3,1,r)+3.*J(3,1,r))).*Q(2,1).*(beta(2, ...
  1).*Q(2,1)).^(-3/2).*Q(3,1).*(beta(3,1).*Q(3,1)).^(-3/2);
Kzz{9,14}=@(r) (-1/4).*kappa.*pi.^2.*q2.*(J(2,1,r).*(2.*r.*dJ(3,1,r)+(-6).*J(3,1, ...
  r))+r.*dJ(2,1,r).*(r.*dJ(3,1,r)+9.*J(3,1,r))).*Q(2,1).*(beta(2,1) ...
  .*Q(2,1)).^(-3/2).*Q(3,1).*(beta(3,1).*Q(3,1)).^(-3/2);
Kzz{9,15}=@(r) (1/4).*kappa.*pi.^2.*q2.*(3.*r.*dJ(2,1,r).*(r.*dJ(3,1,r)+J(3,1,r)) ...
  +J(2,1,r).*((-2).*r.*dJ(3,1,r)+6.*J(3,1,r))).*Q(2,1).*(beta(2,1).* ...
  Q(2,1)).^(-3/2).*Q(3,1).*(beta(3,1).*Q(3,1)).^(-3/2);
Kzz{10,13}=@(r) (1/4).*kappa.*pi.^2.*q2.*(6.*J(2,1,r).*(r.*dJ(3,1,r)+J(3,1,r))+r.* ...
  dJ(2,1,r).*((-1).*r.*dJ(3,1,r)+3.*J(3,1,r))).*Q(2,1).*(beta(2,1).* ...
  Q(2,1)).^(-3/2).*Q(3,1).*(beta(3,1).*Q(3,1)).^(-3/2);
Kzz{10,16}=@(r) (1/4).*kappa.*pi.^2.*q2.*(r.*dJ(2,1,r).*(r.*dJ(3,1,r)+(-3).*J(3,1, ...
  r))+2.*J(2,1,r).*(r.*dJ(3,1,r)+9.*J(3,1,r))).*Q(2,1).*(beta(2,1).* ...
  Q(2,1)).^(-3/2).*Q(3,1).*(beta(3,1).*Q(3,1)).^(-3/2);
Kzz{11,13}=@(r) (1/2).*2.^(-1/2).*kappa.*pi.^2.*q2.*r.*dJ(0,2,r).*(r.*dJ(3,1,r)+ ...
  3.*J(3,1,r)).*Q(0,2).*(beta(0,2).*Q(0,2)).^(-3/2).*Q(3,1).*(beta( ...
  3,1).*Q(3,1)).^(-3/2);
Kzz{11,16}=@(r) (1/2).*2.^(-1/2).*kappa.*pi.^2.*q2.*r.*dJ(0,2,r).*(r.*dJ(3,1,r)+ ...
  3.*J(3,1,r)).*Q(0,2).*(beta(0,2).*Q(0,2)).^(-3/2).*Q(3,1).*(beta( ...
  3,1).*Q(3,1)).^(-3/2);
Kzz{12,14}=@(r) (-1/2).*2.^(-1/2).*kappa.*pi.^2.*q2.*r.*dJ(0,2,r).*(r.*dJ(3,1,r)+ ...
  3.*J(3,1,r)).*Q(0,2).*(beta(0,2).*Q(0,2)).^(-3/2).*Q(3,1).*(beta( ...
  3,1).*Q(3,1)).^(-3/2);
Kzz{12,15}=@(r) (1/2).*2.^(-1/2).*kappa.*pi.^2.*q2.*r.*dJ(0,2,r).*(r.*dJ(3,1,r)+ ...
  3.*J(3,1,r)).*Q(0,2).*(beta(0,2).*Q(0,2)).^(-3/2).*Q(3,1).*(beta( ...
  3,1).*Q(3,1)).^(-3/2);


%--------------------------------------------------------------------------
% calculate the coupling matrix
%--------------------------------------------------------------------------

nf=length(obj.wavelength);
CM=zeros(dims, dims, nf);
I=zeros(dims*(dims+1)/2, 2);

% radial coordinate over which the numerical integration is performed
rho=unique([linspace(0, 1.5*obj.radius, 50) linspace(1.5*obj.radius, 5*obj.radius, 16)])';


% NOTE: diagonal elements (is not zero) have been already 
%       integrated analitically!!!

% cycle over matrix elements
c=0;
for n=1:dims
	for m=n:dims
		
		% true if the element is not zero
		nonzero=false;
		
		% calculate the transverse components
		if any(strcmp(obj.bending.components, {'all', 'transverse'}))
            % longitudinal strain
			if ~isempty(Kzt{indx(n),indx(m)})
				% there is something to calculate
                % performs the integration with respect to r
				CM(n,m,:)=trapz(rho, Kzt{indx(n),indx(m)}(rho));
				nonzero=true;
			elseif ~isempty(Kzt{indx(m),indx(n)})
				% we have to calculate the transpose conjugate
                error('this should never happen!')
				CM(n,m,:)=conj(Kzt{indx(m),indx(n)}); 
				nonzero=true;
            end
            % transverse strain
            if ~isempty(Ktt{indx(n),indx(m)})
				% there is something to calculate
                % these have been already integrated
                CM(n,m,:)=CM(n,m,:)+Ktt{indx(n),indx(m)};
                nonzero=true;
            end
		end
		
 		% calculate the longitudinal components
		if any(strcmp(obj.bending.components, {'all', 'longitudinal'}))	
			% transverse strain
            if ~isempty(Ktz{indx(n),indx(m)})
				% there is something to calculate
				% performs the integration with respect to r avoiding NaN
                rhotmp=rho;
                while 1
                    funz=Ktz{indx(n),indx(m)}(rhotmp);
                    jnan=find(isnan(funz));
                    if isempty(jnan)
                        break
                    else
                        % changes slightly the critical radial coordinates
                        rhotmp(jnan)=rhotmp(jnan)+obj.radius/1e4;
                    end
                end
				CM(n,m,:)=CM(n,m,:)+trapz(rhotmp, funz);
				nonzero=true;
            end
			% longitudinal strain
            if ~isempty(Kzz{indx(n),indx(m)})
				% there is something to calculate
				% performs the integration with respect to r avoiding NaN
                rhotmp=rho;
                while 1
                    funz=Kzz{indx(n),indx(m)}(rhotmp);
                    jnan=find(isnan(funz));
                    if isempty(jnan)
                        break
                    else
                        % changes slightly the critical radial coordinates
                        rhotmp(jnan)=rhotmp(jnan)+obj.radius/1e4;
                    end
                end
				CM(n,m,:)=CM(n,m,:)+trapz(rhotmp, funz);
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

