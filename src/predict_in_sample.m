function yhat = predict_in_sample(y, s, coef_in)
% PREDICT_IN_SAMPLE  one-step-ahead in-sample predictions yhat for t = N+1..T
%   yhat = predict_in_sample(y, s, coef_in)
%
% coef_in may be a struct created by select_model with fields:
%   N, K, vec, c, d, a, alpha, beta
% The code is robust: it uses named fields if present, otherwise unpacks vec.

T = length(y);

% Validate and unpack coef_in
if isstruct(coef_in)
    % Prefer named fields if present
    if isfield(coef_in, 'N') && isfield(coef_in, 'K')
        N = coef_in.N;
        K = coef_in.K;
    else
        error('predict_in_sample: coef struct must include fields N and K.');
    end

    if isfield(coef_in, 'c') && isfield(coef_in, 'd')
        c = coef_in.c;
        d = coef_in.d;
    elseif isfield(coef_in, 'vec')
        tmp = coef_in.vec(:);
        c = tmp(1);
        d = tmp(2);
    else
        error('predict_in_sample: coef struct must include either {c,d} or vec.');
    end

    if isfield(coef_in, 'a')
        a = coef_in.a(:);
    elseif isfield(coef_in, 'vec')
        tmp = coef_in.vec(:);
        if N>0
            a = tmp(3:2+N);
        else
            a = zeros(0,1);
        end
    else
        a = zeros(max(0,N),1);
    end

    if isfield(coef_in, 'alpha')
        alpha = coef_in.alpha(:);
    elseif isfield(coef_in, 'vec') && K>0
        tmp = coef_in.vec(:);
        alpha = tmp(2+N+1 : 2+N+K);
    else
        alpha = zeros(0,1);
    end

    if isfield(coef_in, 'beta')
        beta_sin = coef_in.beta(:);
    elseif isfield(coef_in, 'vec') && K>0
        tmp = coef_in.vec(:);
        beta_sin = tmp(2+N+K+1 : 2+N+2*K);
    else
        beta_sin = zeros(0,1);
    end
else
    error('predict_in_sample: coef_in must be a struct (returned by select_model).');
end

% Basic check
p_expected = 2 + N + 2*K;
% optional: check length of vec if present
if isfield(coef_in,'vec') && ~isempty(coef_in.vec) && length(coef_in.vec) ~= p_expected
    warning('predict_in_sample: coef.vec length mismatch (expected %d)', p_expected);
end

% Prepare output
yhat = zeros(T - N, 1);
rows = (N+1) : T;

for idx = 1:length(rows)
    t = rows(idx);
    val = c + d * t;
    % lagged terms
    for i = 1:N
        val = val + a(i) * y(t - i);
    end
    % seasonal harmonics
    for k = 1:K
        val = val + alpha(k) * cos(2*pi*k*t/s) + beta_sin(k) * sin(2*pi*k*t/s);
    end
    yhat(idx) = val;
end

end
