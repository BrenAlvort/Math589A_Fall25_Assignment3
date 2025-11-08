function [A,b,meta] = build_design(y, s, N, K)
% BUILD_DESIGN  Construct LS system for N-th order difference eq + K harmonics.
%   y : Tx1 double (column)
%   s : seasonal period (e.g., 12)
%   N : nonnegative integer (order of past)
%   K : nonnegative integer (# harmonics)
% Returns:
%   A : (T-N)x(2+N+2K) matrix  [1, t, lags..., cos..., sin...]
%   b : (T-N)x1 vector         [y_{N+1: T}]
%   meta : struct with fields: .rows=M, .p=p, .t=(N+1:T).'

    y = y(:);
    T = numel(y);
    M = T - N;
    p = 2 + N + 2*K;            % <-- include both c and d (constant and time)
    if M <= p
        error('Underdetermined: T-N (= %d) must exceed p (= %d).', M, p);
    end

    b = y(N+1:T);
    A = zeros(M, p);

    t = (N+1:T).';

    % column 1: constant
    A(:,1) = 1;
    % column 2: time t for linear trend d * t
    A(:,2) = t;

    col = 2;
    % lag columns y_{t-1} ... y_{t-N}
    for i = 1:N
        col = col + 1;
        A(:, col) = y(N+1 - i : T - i);
    end

    % cosine columns
    for k = 1:K
        col = col + 1;
        A(:, col) = cos(2*pi*k*t/s);
    end

    % sine columns
    for k = 1:K
        col = col + 1;
        A(:, col) = sin(2*pi*k*t/s);
    end

    meta = struct('rows', M, 'p', p, 't', t);
end
