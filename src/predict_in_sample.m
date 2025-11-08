function yhat = predict_in_sample(y, s, coef_in, varargin)
T = length(y);

% unwrap a wrapper that holds .coef
if isstruct(coef_in) && isfield(coef_in,'coef') && ~isfield(coef_in,'vec') && ~isfield(coef_in,'c')
    coef_in = coef_in.coef;
end

% parse coef (robust)
[N,K,c,d,a,alpha,beta_sin] = robust_parse_coef(coef_in, varargin{:});

% ensure shapes and padding
a = reshape(a,[],1); alpha = reshape(alpha,[],1); beta_sin = reshape(beta_sin,[],1);
if numel(a) < N, a(end+1:N,1) = 0; end
if numel(alpha) < K, alpha(end+1:K,1) = 0; end
if numel(beta_sin) < K, beta_sin(end+1:K,1) = 0; end

yhat = zeros(T - N, 1);
rows = (N+1):T;
for i = 1:numel(rows)
    t = rows(i);
    val = c + d * t;
    for j = 1:N
        val = val + a(j) * y(t - j);
    end
    for k = 1:K
        val = val + alpha(k) * cos(2*pi*k*t/s) + beta_sin(k) * sin(2*pi*k*t/s);
    end
    yhat(i) = val;
end
end

function [N,K,c,d,a,alpha,beta_sin] = robust_parse_coef(coef_in, varargin)
% Return canonical N,K,c,d,a,alpha,beta
N = 0; K = 0; c = []; d = []; a = zeros(0,1); alpha = zeros(0,1); beta_sin = zeros(0,1);

if isnumeric(coef_in)
    if numel(varargin) < 2, error('predict_in_sample: numeric coef requires N,K'); end
    N = varargin{1}; K = varargin{2};
    tmp = coef_in(:);
    if numel(tmp) ~= 2 + N + 2*K, error('predict_in_sample: coef length mismatch'); end
    c = tmp(1); d = tmp(2);
    if N>0, a = tmp(3:2+N); end
    if K>0
        alpha = tmp(2+N+1:2+N+K);
        beta_sin = tmp(2+N+K+1:2+N+2*K);
    end
    return;
end

if isstruct(coef_in)
    % direct fields
    if isfield(coef_in,'N'), N = coef_in.N; end
    if isfield(coef_in,'K'), K = coef_in.K; end
    if isfield(coef_in,'c') && isfield(coef_in,'d'), c = coef_in.c; d = coef_in.d; end
    if isfield(coef_in,'a'), a = coef_in.a(:); end
    if isfield(coef_in,'alpha'), alpha = coef_in.alpha(:); end
    if isfield(coef_in,'beta'), beta_sin = coef_in.beta(:); end
    if isfield(coef_in,'vec') && ~isempty(coef_in.vec)
        tmp = coef_in.vec(:);
        p = numel(tmp);
        c = tmp(1); d = tmp(2);
        K = floor((p-2)/2);
        N = p - 2 - 2*K;
        if N>0, a = tmp(3:2+N); end
        if K>0
            alpha = tmp(2+N+1:2+N+K);
            beta_sin = tmp(2+N+K+1:2+N+2*K);
        end
        return;
    end

    % fallback: find largest numeric vector field (if any)
    fn = fieldnames(coef_in);
    bestlen = 0; bestvec = [];
    for i=1:numel(fn)
        val = coef_in.(fn{i});
        if isnumeric(val) && numel(val) > bestlen
            bestlen = numel(val); bestvec = val(:);
        end
    end
    if bestlen >= 2
        tmp = bestvec; p = numel(tmp);
        c = tmp(1); d = tmp(2);
        K = floor((p-2)/2);
        N = p - 2 - 2*K;
        if N>0, a = tmp(3:2+N); end
        if K>0
            alpha = tmp(2+N+1:2+N+K);
            beta_sin = tmp(2+N+K+1:2+N+2*K);
        end
        return;
    end
end

if isempty(c) || isempty(d)
    error('predict_in_sample: coef struct must include either fields c,d or vec.');
end
end
