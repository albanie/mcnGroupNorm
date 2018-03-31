function plotResults(results)

opts.enforceAxis = false ;
% set y limits to prevent matlab following anomolies
if opts.enforceAxis
  if results(1).batchSize == 4
    ylims = [ 0 0.3 ; 0 0.1 ] ;
  else
    ylims = [ 0 0.1 ; 0 0.05 ] ;
  end
end

figure(1) ; clf ;
subplot(1,2,1) ;
hold all ;
styles = {'o-', '+--', '+-', '.-'} ;
keyboard
for i = 1:numel(results)
  semilogy([results(i).info.val.objective]', styles{i}) ;
end
keyboard
xlabel('Training samples [x 10^3]') ; ylabel('energy') ;
grid on ;
h = legend(results(:).name) ;
set(h,'color','none');
batchSize = results(1).batchSize ;
title(sprintf('objective-(bs%d)', batchSize)) ;
if opts.enforceAxis
  ylim(ylims(1,:)) ;
end
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
if opts.enforceAxis
  ylim(ylims(2,:)) ;
end
drawnow ;

% this is a function for plotting figures in the terminal
% (the function can be found at https://github.com/albanie/zvision)
% but can be commented out if you are using a normal GUI
if exist('zs_dispFig', 'file'), zs_dispFig ; end
