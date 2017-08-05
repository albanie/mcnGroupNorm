function test_mcnBReNorm
% run tests for BReNorm module

  % add tests to path
  addpath(fullfile(fileparts(mfilename('fullpath')), 'matlab/xtest')) ;
  addpath(fullfile(vl_rootnn, 'matlab/xtest/suite')) ;

  % test network layers
  run_brenorm_tests('command', 'nn') ;
