addpath(fullfile('..','src'));

y = (1:20)'; s = 12; N = 3; K = 2;
[A,b,meta] = build_design(y,s,N,K);

% sizes
assert(size(A,1) == 20 - N);
assert(size(A,2) == 2 + N + 2*K);      % intercept + trend + N lags + 2K seasonals
assert(all(b(1:3) == y(N+1:N+3)));     % b starts at y_{N+1} = y(4)

% first row content (t = N+1)
t1  = meta.t(1);
row = A(1,:);
assert(row(1) == 1);                   % intercept
assert(row(2) == t1);                  % trend column equals t

% lags y_{t-1},...,y_{t-N} appear next, in that order
lags = row(3 : 3+N-1);
assert(all(lags == [y(t1-1), y(t1-2), y(t1-3)]));   % [y3,y2,y1] when t1=4

% seasonal blocks exist with K columns each
cos_blk = row(3+N : 3+N+K-1);
sin_blk = row(3+N+K : 3+N+2*K-1);
assert(numel(cos_blk) == K && numel(sin_blk) == K);

disp('build_design OK');
