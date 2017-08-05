function mnist_renorm_experiment1
%MNIST_RENORM_EXPERIMENT1 
% A set of simple experiments on mnist with batch renormalization
%
% batch renomalization is explained in the paper:
%   Ioffe, Sergey. "Batch Renormalization: Towards Reducing Minibatch 
%   Dependence in Batch-Normalized Models." arXiv preprint 
%   arXiv:1702.03275 (2017).
%
% Copyright (C) 2017 Samuel Albanie
% All rights reserved.

  warmupEpochs = 5 ;
  transitionEpochs = 5 ;
  lastEpochs = 20 ;
  rStart = 1 ; rEnd = 3 ;
  dStart = 0 ; dEnd = 5 ;
  rSteady = rStart:(rEnd-rStart) / transitionEpochs: rEnd ;
  dSteady = dStart:(dEnd-dStart) / transitionEpochs: dEnd ;
  warmup = repmat([rStart dStart], warmupEpochs, 1) ;
  steady = [ rSteady' dSteady' ] ;
  last = repmat([rEnd dEnd], lastEpochs, 1) ;
  clips = [ warmup ; steady ; last ] ;


  results_big = runner(256, clips) ;
  results_medium = runner(128, clips) ;
  results_small = runner(64, clips) ;
  results_tiny = runner(32, clips) ;
  results_mini = runner(16, clips) ;
  results_v_mini = runner(4, clips) ;

  plotResults(results_big)  ;
  plotResults(results_medium)  ;
  plotResults(results_small)  ;
  plotResults(results_tiny)  ;
  plotResults(results_v_mini)  ;
  plotResults(results_mini)  ;

% ------------------------------------------
function results = runner(batchSize, clips)
% -----------------------------------------
  train.gpus = 4 ;
  train.continue = 1 ;
  train.numEpochs = 30 ;
  train.numEpochs = size(clips, 1) - 1 ;
  train.batchSize = batchSize ;
  expRoot = fullfile(vl_rootnn, 'data/mnist-exps/exp1') ;
  results = struct() ;

  opts = {{'train', train} , ...
          {'train', train, 'batchNormalization', 1}, ...
          {'train', train, 'batchRenormalization', 1, ...
           'clips', clips, 'alpha', 0.01}, ...
           } ;
  names = {'BSLN', 'BNORM', 'RENORM'} ;
  if batchSize <= 32
    opts(1) = [] ;
    names(1) = [] ;
  end
  for i = 1:numel(names)
    expOpts = opts{i} ;
    [~, info] = mnist_renorm(expOpts{:}, 'expRoot', expRoot) ;
    results(i).info = info ;
    results(i).batchSize = batchSize ;
    results(i).name = names{i} ;
  end
