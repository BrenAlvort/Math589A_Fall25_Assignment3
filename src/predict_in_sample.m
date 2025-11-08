function yhat = predict_in_sample(y, s, coef_in, varargin)
% Robust one-step in-sample predictions. Accepts many coef formats.

T = length(y);

% Try to find usable coef inside wrappers
[ N, K, c, d, a, alpha, beta_sin ] = parse_coef(coef_in, varargin{:});

% Ensure correct shapes
a = reshape(a, [], 1);
alpha = reshape(alpha, [], 1);
beta_sin = reshape(beta_sin, [], 1);
if numel(a) < N, a = [a; zeros(N - numel(a),1)]; end
if numel(alpha) < K, alpha = [alpha; zeros(K - numel(alpha),1)]; end
if numel(beta_sin) < K, beta_sin = [beta_sin; zeros(K - numel(beta_sin),1)]; end

% Build predictions
yhat = zeros(T - N, 1);
rows = (N+1):T;
for idx = 1:length(rows)
    t = rows(idx);
    val = c + d * t;
    for i = 1:N
        val = val + a(i) * y(t - i);
    end
    for k = 1:K
        val = val + alpha(k) * cos(2*pi*k*t/s) + beta_sin(k) * sin(2*pi*k*t/s);
    end
    yhat(idx) = val;
end

end

function [N,K,c,d,a,alpha,beta_sin] = parse_coef(coef_in, varargin)
% parse_coef: robustly extract N,K and parameters from variety of coef formats

% default
N = 0; K = 0; c = 0; d = 0; a = zeros(0,1); alpha = zeros(0,1); beta_sin = zeros(0,1);

% unwrap common wrapper
if isstruct(coef_in) && isfield(coef_in,'coef') && ~isfield(coef_in,'vec') ...
        && ~isfield(coef_in,'c') && ~isfield(coef_in,'d')
    coef_in = coef_in.coef;
end

% numeric vector + N,K in varargin
if isnumeric(coef_in)
    if numel(varargin) < 2, error('predict_in_sample: numeric coef needs N and K'); end
    N = varargin{1}; K = varargin{2};
    tmp = coef_in(:);
    if length(tmp) ~= 2 + N + 2*K, error('predict_in_sample: coef vec length mismatch'); end
    c = tmp(1); d = tmp(2);
    if N>0, a = tmp(3:2+N); else a = zeros(0,1); end
    if K>0
        alpha = tmp(2+N+1 : 2+N+K);
        beta_sin = tmp(2+N+K+1 : 2+N+2*K);
    end
    return;
end

% coef_in is struct: try direct fields
if isstruct(coef_in)
    % direct N,K
    if isfield(coef_in,'N'), N = coef_in.N; end
    if isfield(coef_in,'K'), K = coef_in.K; end

    % direct c,d
    if isfield(coef_in,'c') && isfield(coef_in,'d')
        c = coef_in.c; d = coef_in.d;
    end

    % vec
    if isfield(coef_in,'vec') && ~isempty(coef_in.vec)
        tmp = coef_in.vec(:);
        p = numel(tmp);
        if p >=2
            c = tmp(1); d = tmp(2);
            if isempty(N) || isempty(K) || (2 + N + 2*K ~= p)
                % infer N,K
                K = floor((p-2)/2);
                N = p - 2 - 2*K;
            end
            if N>0, a = tmp(3:2+N); end
            if K>0
                alpha = tmp(2+N+1 : 2+N+K);
                beta_sin = tmp(2+N+K+1 : 2+N+2*K);
            end
            return;
        end
    end

    % named a/alpha/beta fields
    if isfield(coef_in,'a'), a = coef_in.a(:); N = numel(a); end
    if isfield(coef_in,'alpha'), alpha = coef_in.alpha(:); K = numel(alpha); end
    if isfield(coef_in,'beta'), beta_sin = coef_in.beta(:); K = max(K, numel(beta_sin)); end

    % if we still don't have c,d but there exists a numeric field that looks like vec, try to find it
    f = fieldnames(coef_in);
    bestlen = 0; bestvec = [];
    for i=1:numel(f)
        val = coef_in.(f{i});
        if isnumeric(val) && numel(val) > bestlen
            bestlen = numel(val);
            bestvec = val(:);
        end
    end
    if bestlen >= 2 && isempty(coef_in) == 0
        tmp = bestvec;
        p = numel(tmp);
        c = tmp(1); d = tmp(2);
        K = floor((p-2)/2);
        N = p - 2 - 2*K;
        if N>0, a = tmp(3:2+N); end
        if K>0
            alpha = tmp(2+N+1 : 2+N+K);
            beta_sin = tmp(2+N+K+1 : 2+N+2*K);
        end
        return;
    end
end

% Final fallback: require c and d exist
if isempty(c) || isempty(d)
    error('predict_in_sample: coef struct must include either fields c,d or vec.');
end
end
