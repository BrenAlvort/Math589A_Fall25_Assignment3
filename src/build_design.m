function [A,b,meta] = build_design(y, s, N, K)
% BUILD_DESIGN  Construct LS system for N-th order difference eq + K harmonics.
%   y : Tx1 double (column)
%   s : scalar season (e.g., 12)
%   N : nonnegative integer (order of past)
%   K : nonnegative integer (# harmonics)
% Returns:
%   A : (T-N) x (2 + N + 2K) matrix  [1, t, lags..., cos..., sin...]
%   b : (T-N) x 1 vector             [y_{N+1:T}]
%   meta : struct with fields: .rows=M, .p=p, .t=(N+1:T).'
%
    narginchk(4,4);
    y = y(:);
    T = numel(y);
    if N < 0 || K < 0 || floor(N)~=N || floor(K)~=K
        error('N and K must be nonnegative integers.');
    end

    M = T - N;
    p = 2 + N + 2*K;
    if M < p
        error('Underdetermined: T-N (= %d) must be >= p (= %d).', M, p);
    end

    % response and time index
    b = y(N+1:T);
    t = (N+1:T).';

    % preallocate A
    A = zeros(M, p);

    col = 1;
    A(:, col) = 1;           % intercept
    col = col + 1;
    A(:, col) = t;           % linear trend (absolute time)
    col = col + 1;

    % lag columns y_{t-1},...,y_{t-N}
    for i = 1:N
        A(:, col) = y(N+1-i : T-i);
        col = col + 1;
    end

    % vectorized seasonal harmonics if K > 0
    if K > 0
        kvec = (1:K);
        % M x K cosine and sine matrices
        % use outer product t * kvec to avoid loops
        TK = (t * kvec) * (2*pi/s);        % M x K matrix of (2*pi*k*t/s)
        A(:, col:col+K-1) = cos(TK);       % cos columns
        col = col + K;
        A(:, col:col+K-1) = sin(TK);       % sin columns
        col = col + K;
    end

    meta = struct('rows', M, 'p', p, 't', t);
end
