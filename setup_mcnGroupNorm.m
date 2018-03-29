function setup_mcnGroupNorm
%SETUP_MCNGROUPNORM Sets up mcnGroupNorm by adding its folders
% to the MATLAB path
%
% Copyright (C) 2018 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/example']) ;
