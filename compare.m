% init

C = 3 ;
x = randn(5,5,C,2) ;
g = rand(1,C) ;
b = rand(1,C) ;

% ------------------------------------------------------------------
%                                                     Training mode
% ------------------------------------------------------------------

y1 = vl_nnbnorm(x, g, b) ;
y2 = vl_nnbnorm2(x, g, b) ;

r1 = norm(y1(:)) ;
r2 = norm(y2(:)) ;

diff = y2 - y1 ;

fprintf('res bnorm - size: [%d,%d,%d,%d], norm: %f\n', size(y1), r1) ;
fprintf('res bnorm - size: [%d,%d,%d,%d], norm: %f\n', size(y2), r2) ;
fprintf('diff norm: %f\n', norm(diff(:))) ;

% ------------------------------------------------------------------
%                                     Training mode fetching moments
% ------------------------------------------------------------------

[y1, moments1] = vl_nnbnorm(x, g, b) ;
[y2, moments2] = vl_nnbnorm2(x, g, b) ;

r1 = norm(y1(:)) ;
r2 = norm(y2(:)) ;
m1 = norm(moments1(:)) ;
m2 = norm(moments2(:)) ;

diff = y2 - y1 ;
momentDiff = m2 - m1 ;

fprintf('diff norm: %f\n', norm(diff(:))) ;
fprintf('diff moments: %f\n', norm(momentDiff(:))) ;

% ------------------------------------------------------------------
%                                            Training backwards mode
% ------------------------------------------------------------------

% check ders
dzdy = rand(size(x)) ;
[dzdx1, dzdg1, dzdb1] = vl_nnbnorm(x, g, b, dzdy) ;
[dzdx2, dzdg2, dzdb2] = vl_nnbnorm2(x, g, b, dzdy) ;

ddiff = dzdx2 - dzdx1 ;
fprintf('diff der: %f\n', norm(ddiff(:))) ;

% ------------------------------------------------------------------
%                                                       Testing mode
% ------------------------------------------------------------------

% in test mode, moments are used
moments = rand(C, 2) ;

ym1 = vl_nnbnorm(x, g, b, 'moments', moments) ;
ym2 = vl_nnbnorm2(x, g, b, 'moments', moments) ;

r1 = norm(ym1(:)) ;
r2 = norm(ym2(:)) ;

mdiff = ym2 - ym1 ;

fprintf('moments diff norm: %f\n', norm(mdiff(:))) ;

% -------------------------------------------------------------------
%                                         Backwards mode with moments
% ---------------------------------=---------------------------------

% check ders
dzdy = rand(size(x)) ;
[dzdx1, dzdg1, dzdb1] = vl_nnbnorm(x, g, b, dzdy) ;
[dzdx2, dzdg2, dzdb2] = vl_nnbnorm2(x, g, b, dzdy) ;

ddiff = dzdx2 - dzdx1 ;
fprintf('diff der: %f\n', norm(ddiff(:))) ;
