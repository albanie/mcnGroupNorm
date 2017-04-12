function [y, dzdg, dzdb, moments] = vl_nnbrenorm(x, g, b, clips, varargin) 
%VL_NNBRENORM CNN batch renormalisation.
%   Y = VL_NNBRENORM(X,G,B,R,D) applies batch renormalization to the input
%   X. Batch renormalization is defined as:
%
%      Y(i,j,k,t) = G(k) * X_HAT(i,j,k,t) + B(k)
%
%   where
%      X_HAT(i,j,k,t) = R(k) * (X_HAT(i,j,k,t) - mu(k)) / sigma(k) + D(k)
%
%      mu(k) = mean_ijt X(i,j,k,t),
%      sigma2(k) = mean_ijt (X(i,j,k,t) - mu(k))^2,
%      sigma(k) = sqrt(sigma2(k) + EPSILON)
%      R(k) = clip(sigma(k) / moments(2,k)), [1/rMax, rMax])
%      D(k) = clip((mu(k) - moments(1,k))/ moments(2,k)), [-dMax, dMax])
%
%   and we define clip(x, [a b]) to be the operation that clips the value
%   of x to lie inside the range [a b].
%   are respectively the per-channel mean, variance, and standard
%   deviation of each feature channel in the data X. The parameters
%   G(k) and B(k) are multiplicative and additive constants use to
%   scale each data channel, while R(k) and D(k) are used to balance the 
%   current estimate of feature means and variances between the statistics
%   gathered from the current mini-batch, and rolling averages over previous
%   mini-batches, as discussed in the paper:
%
%  `Batch Renormalization: Towards Reducing Minibatch Dependence in
%   Batch-Normalized Models` by Sergey Ioffe, 2017

opts.moments = [] ;
[opts, dzdy] = vl_argparsepos(opts, varargin) ;

moments = opts.moments ;

% unpack parameters
epsilon = 1e-4 ;
rMax = clips(1) ;
dMax = clips(2) ;

rolling_mu = permute(moments(:,1), [3 2 1]) ;
rolling_sigma = permute(moments(:,2), [3 2 1]) ;

% first compute the statistics per channel for the current 
% minibatch and normalize
mu = chanAvg(x) ;
sigma2 = chanAvg(bsxfun(@minus, x, mu).^ 2) ;
sigma = sqrt(sigma2 + epsilon) ;
x_hat_ = bsxfun(@rdivide, bsxfun(@minus, x, mu), sigma) ;

% then "renormalize"
r = bsxfun(@min, bsxfun(@max, sigma ./ rolling_sigma, 1 / rMax), rMax) ;
d = bsxfun(@min, bsxfun(@max,(mu - rolling_mu)./rolling_sigma, -dMax), dMax) ;
x_hat = bsxfun(@plus, bsxfun(@times, x_hat_, r), d) ;

if isempty(dzdy)
	% apply gain
	res = bsxfun(@times, g, x_hat) ;

	% add bias
	y = bsxfun(@plus, res, b) ;
else
  % precompute some common terms 
  t1 = bsxfun(@minus, x, mu) ;
  t2 = bsxfun(@rdivide, 1, sqrt(sigma2 + epsilon)) ;
  t3 = bsxfun(@rdivide, r, sigma2) ;
  sz = size(x) ; m = prod([sz(1:2) sz(4)]) ;
  dzdy = dzdy{1} ;

  dzdx_hat = bsxfun(@times, dzdy, g) ;
  dzdsigma = chanSum(dzdx_hat .* bsxfun(@times, t1, -t3)) ;
  dzdmu = chanSum(bsxfun(@times, dzdx_hat, -t3)) ;

  t4 = bsxfun(@times, dzdx_hat, t3) + ...
       bsxfun(@times, dzdsigma,  bsxfun(@rdivide, t1, m * sigma)) ;
  dzdx = bsxfun(@plus, t4, dzdmu * (1/m)) ;
                                    
  y = dzdx ;
  dzdg = chanSum(dzdx_hat .* dzdy) ;
  dzdb = chanSum(dzdy) ;
end

% compute moments
if nargout == 2
		moments = horzcat(squeeze(mu), squeeze(sigma)) ;
    dzdg = moments ;
end

% ------------------------
function avg = chanAvg(x)
% ------------------------
avg = mean(mean(mean(x, 1), 2), 4) ;

% ------------------------
function res = chanSum(x)
% ------------------------
res = sum(sum(sum(x, 1), 2), 4) ;
