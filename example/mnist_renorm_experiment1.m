function mnist_renorm_experiment1
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

warmupEpochs = 5 ;
transitionEpochs = 5 ;
lastEpochs = 30 ;
rStart = 1 ; rEnd = 3 ;
dStart = 0 ; dEnd = 5 ;
rSteady = rStart:(rEnd-rStart) / transitionEpochs: rEnd ;
dSteady = dStart:(dEnd-dStart) / transitionEpochs: dEnd ;
warmup = repmat([rStart dStart], warmupEpochs, 1) ;
steady = [ rSteady' dSteady' ] ;
last = repmat([rEnd dEnd], lastEpochs, 1) ;
clips = [ warmup ; steady ; last ] ;

train.continue = 1 ;
train.numEpochs = 40 ;

results_big = runner(256, clips) ;
results_mid = runner(128, clips) ;
results_small = runner(64, clips) ;
results_tiny = runner(32, clips) ;
results_mini = runner(16, clips) ;

plotResults(results_big)  ;
plotResults(results_medium)  ;
plotResults(results_small)  ;
plotResults(results_tiny)  ;
plotResults(results_mini)  ;
zv_dispFig ;

% ------------------------------------------
function results = runner(batchSize, clips)
% -----------------------------------------
results = struct() ;
train.continue = 1 ;
train.gpus = [1] ;
train.numEpochs = size(clips, 1) - 1 ;
train.batchSize = batchSize ;
expRoot = fullfile(vl_rootnn, 'data/mnist-exps/exp1') ;
opts = {{'train', train} , ...
        {'train', train, 'batchNormalization', 1}, ...
        {'train', train, 'batchRenormalization', 1, ...
         'clips', clips, 'alpha', 0.01}} ;
names = {'BSLN', 'BNORM', 'RENORM'} ;
for i = 1:numel(names)
  expOpts = opts{i} ;
  [~, info] = mnist_renorm(expOpts{:}, 'expRoot', expRoot) ;
  results(i).info = info ;
  results(i).batchSize = batchSize ;
  results(i).name = names{i} ;
end

% ---------------------------
function plotResults(results)
% ---------------------------
figure(1) ; clf ;
subplot(1,2,1) ;
hold all ;
styles = {'o-', '+--', '+-'} ;
for i = 1:numel(results)
  semilogy([results(i).info.val.objective]', styles{i}) ; 
end
xlabel('Training samples [x 10^3]') ; ylabel('energy') ;
grid on ;
h = legend(results(:).name) ;
set(h,'color','none');
batchSize = results(1).batchSize ;
title(sprintf('objective-(bs%d)', batchSize)) ;
subplot(1,2,2) ;
hold all ;
for i = 1:numel(results)
  plot([results(i).info.val.error]',styles{i}) ;
end
h = legend(results(:).name) ;
grid on ;
xlabel('Training samples [x 10^3]'); ylabel('error') ;
set(h,'color','none') ;
title(sprintf('error-(bs%d)', batchSize)) ;
drawnow ;
