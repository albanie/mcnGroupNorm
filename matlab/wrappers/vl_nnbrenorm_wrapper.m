function [y, dzdg, dzdb, moments] = vl_nnbrenorm_wrapper(x, g, b, ...
                                              moments, clips, test, varargin)
%VL_NNBNORM_WRAPPER AutoNN wrapper for MatConvNet's vl_nnbrenorm
%   VL_NNBNORM has a non-standard interface (returns a derivative for the
%   moments, even though they are not an input), so we must wrap it.
%   Layer.vl_nnbrenorm replaces a standard VL_NNBNORM call with this one.
%
%   This also lets us supports nice features like setting the parameter
%   sizes automatically (e.g. building a net with VL_NNBRENORM(X) is valid).

% Copyright (C) 2016-2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  % use number of channels in X to extend scalar (i.e. default) params to
  % the correct size. this way the layer can be constructed without
  % knowledge of the number of channels. scalars also permit gradient
  % accumulation with any tensor shape (in CNN_TRAIN_AUTONN).

  [opts, dzdy] = vl_argparsepos(struct(), varargin) ;

  if isscalar(g)
    g(1,1,1:size(x,3)) = g ;
  end
  if isscalar(b)
    b(1,1,1:size(x,3)) = b ;
  end
  if isscalar(moments)
    moments(1:size(x,3),1:2) = moments ;
  end

  if isempty(dzdy)
    y = vl_nnbrenorm(x, g, b, moments, clips, test, varargin{:}) ;
  else
    [y, dzdg, dzdb, moments] = vl_nnbrenorm(x, g, b, moments, clips, ...
                                            test, dzdy{1}, varargin{2:end}) ;
    moments = moments * size(x, 4) ;
  end
end

