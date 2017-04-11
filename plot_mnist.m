function plot_mnist(varargin) 

base = '/Users/samuelalbanie/coding/libs/matconvnets' ;
opts.rootDir = fullfile(base, '/bnorm-matconvnet/data/mnist-exps') ;
opts = vl_argparse(opts, varargin) ;

expDirs = dir(fullfile(opts.rootDir, '*')) ;
f = zv_ignoreSysFiles(expDirs) ;
expDirs = cellfun(@(x) {fullfile(opts.rootDir, x)}, {f.name}) ;

results = extractStats(expDirs) ;
keyboard

% ------------------------------------------
function results = extractStats(expDirs)
% ------------------------------------------
results = struct() ;

for i = 1:numel(expDirs)
  last = findLastCheckpoint(expDirs{i}) ;
  checkpoint = fullfile(expDirs{i}, sprintf('net-epoch-%d.mat', last)) ;
  data = load(checkpoint) ;
  [~,name] = fileparts(expDirs{i}) ;
  results(i).name = name ;
  results(i).train_error = [data.stats.train.error] ;
  results(i).train_obj = [data.stats.train.objective] ;
  results(i).val_error = [data.stats.val.error] ;
  results(i).val_obj = [data.stats.val.objective] ;
end

% ------------------------------------------
function epoch = findLastCheckpoint(expDir)
% ------------------------------------------

list = dir(fullfile(expDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
