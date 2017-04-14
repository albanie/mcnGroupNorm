function mnist_renorm_experiment2
% Some simple experiments on mnist with batch renormalization

% batch renomalization is explained in the paper:
%   Ioffe, Sergey. "Batch Renormalization: Towards Reducing Minibatch 
%   Dependence in Batch-Normalized Models." arXiv preprint 
%   arXiv:1702.03275 (2017).


% ---------------------
% batch renormalization
% ---------------------

% To demonstrate the role of batch renormalization, it is necessary to 
% examine the impact of batch size on batch normalization. In the following,
% we first train a model on mnist with a reasonably large batch size.

warmupEpochs = 10 ;
transitionEpochs = 10 ;
lastEpochs = 30 ;
rStart = 1 ; rEnd = 3 ;
dStart = 0 ; dEnd = 5 ;
rSteady = rStart:(rEnd-rStart) / transitionEpochs: rEnd ;
dSteady = dStart:(dEnd-dStart) / transitionEpochs: dEnd ;
warmup = repmat([rStart dStart], warmupEpochs, 1) ;
steady = [ rSteady' dSteady' ] ;
last = repmat([rEnd dEnd], lastEpochs, 1) ;
clips = [ warmup ; steady ; last ] ;


alphas = [0.1 0.01 0.001] ;

results_big = runner(256, alphas, clips) ;
results_mid = runner(128, alphas, clips) ;
results_small = runner(64, alphas, clips) ;
results_tiny = runner(32, alphas, clips) ;
results_mini = runner(16, alphas, clips) ;

plotResults(results_big)  ;
plotResults(results_medium)  ;
plotResults(results_small)  ;
plotResults(results_tiny)  ;
plotResults(results_mini)  ;

% -------------------------------------------------
function results = runner(batchSize, alphas, clips)
% -------------------------------------------------
results = struct() ;
train.continue = 1 ;
train.gpus = [4] ;
train.numEpochs = size(clips, 1) - 1 ;
train.batchSize = batchSize ;
expRoot = fullfile(vl_rootnn, 'data/mnist-exps/exp2') ;
for i = 1:numel(alphas)
  [~, info] = mnist_renorm('train', train, ...
                           'expRoot', expRoot, ...
                           'batchRenormalization', true, ...
                           'clips', clips, 'alpha', alphas(i)) ;
  results(i).info = info ;
  results(i).batchSize = batchSize ;
  results(i).name = sprintf('alpha: %g', alphas(i)) ;
end
