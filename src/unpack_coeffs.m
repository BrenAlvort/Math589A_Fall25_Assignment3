function coef = unpack_coeffs(beta, N, K)
% UNPACK_COEFFS  helper matching indexing convention:
%   beta = [c; d; a1..aN; alpha1..alphaK; beta1..betaK]
% Returns struct with fields: vec, N, K, c, d, a, alpha, beta
    beta = beta(:);
    p_expected = 2 + N + 2*K;
    % pad or truncate safely
    if numel(beta) < p_expected
        beta = [beta; zeros(p_expected - numel(beta), 1)];
    elseif numel(beta) > p_expected
        beta = beta(1:p_expected);
    end

    coef.vec = beta;
    coef.N = N;
    coef.K = K;
    coef.c = beta(1);
    coef.d = beta(2);

    % safely extract blocks
    idx = 3;
    if N > 0
        coef.a = beta(idx:idx+N-1);
    else
        coef.a = zeros(0,1);
    end
    idx = 3 + N;
    if K > 0
        coef.alpha = beta(idx:idx+K-1);
        coef.beta  = beta(idx+K:idx+2*K-1);
    else
        coef.alpha = zeros(0,1);
        coef.beta  = zeros(0,1);
    end

    % ensure column shapes
    coef.a = coef.a(:);
    coef.alpha = coef.alpha(:);
    coef.beta = coef.beta(:);
end
