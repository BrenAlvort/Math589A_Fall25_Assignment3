function yhat = predict_in_sample(y, s, coef_in, varargin)
T = length(y);

% unwrap wrapper if present
if isstruct(coef_in) && isfield(coef_in,'coef') && ~isfield(coef_in,'vec') && ~isfield(coef_in,'c')
    coef_in = coef_in.coef;
end

% aggressive parsing: try to find c,d,vec,a,alpha,beta in many shapes
[N,K,c,d,a,alpha,beta_sin] = aggressive_parse_coef(coef_in, varargin{:});

% if c or d missing, attempt best-effort default (but print diag)
if isempty(c) || isempty(d)
    fprintf('[predict_in_sample] WARNING: c/d missing. c=%s d=%s. Trying to infer from vec.\n', mat2str(c), mat2str(d));
    if ~isempty(coef_in) && isstruct(coef_in) && isfield(coef_in,'vec')
        tmp = coef_in.vec(:);
        if numel(tmp) >= 2, c = tmp(1); d = tmp(2); end
    end
end

% final padding/truncation safety
a = reshape(a,[],1); alpha = reshape(alpha,[],1); beta_sin = reshape(beta_sin,[],1);
if numel(a) < N, a(end+1:N,1) = 0; end
if numel(alpha) < K, alpha(end+1:K,1) = 0; end
if numel(beta_sin) < K, beta_sin(end+1:K,1) = 0; end

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

function [N,K,c,d,a,alpha,beta_sin] = aggressive_parse_coef(coef_in, varargin)
% returns canonical (N,K,c,d,a,alpha,beta)
N = 0; K = 0; c = []; d = []; a = zeros(0,1); alpha = zeros(0,1); beta_sin = zeros(0,1);

% numeric vector case: need N,K in varargin
if isnumeric(coef_in)
    if numel(varargin) < 2, error('predict_in_sample: numeric coef needs N,K'); end
    N = varargin{1}; K = varargin{2};
    tmp = coef_in(:); c = tmp(1); d = tmp(2);
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
    if isfield(coef_in,'c'), c = coef_in.c; end
    if isfield(coef_in,'d'), d = coef_in.d; end
    if isfield(coef_in,'a'), a = coef_in.a(:); end
    if isfield(coef_in,'alpha'), alpha = coef_in.alpha(:); end
    if isfield(coef_in,'beta'), beta_sin = coef_in.beta(:); end
    if isfield(coef_in,'vec') && ~isempty(coef_in.vec)
        tmp = coef_in.vec(:); p = numel(tmp);
        if isempty(N) || isempty(K) || (2 + N + 2*K ~= p)
            K = floor((p-2)/2); N = p - 2 - 2*K;
        end
        c = tmp(1); d = tmp(2);
        if N>0, a = tmp(3:2+N); end
        if K>0
            alpha = tmp(2+N+1:2+N+K);
            beta_sin = tmp(2+N+K+1:2+N+2*K);
        end
        return;
    end

    % fallback: scan for any numeric vector fields, take the longest
    fn = fieldnames(coef_in);
    bestlen = 0; bestvec = [];
    for i=1:numel(fn)
        val = coef_in.(fn{i});
        if isnumeric(val) && numel(val) > bestlen
            bestlen = numel(val);
            bestvec = val(:);
        end
    end
    if bestlen >= 2
        tmp = bestvec; p = numel(tmp);
        K = floor((p-2)/2); N = p - 2 - 2*K;
        c = tmp(1); d = tmp(2);
        if N>0, a = tmp(3:2+N); end
        if K>0
            alpha = tmp(2+N+1:2+N+K);
            beta_sin = tmp(2+N+K+1:2+N+2*K);
        end
        return;
    end
end

end
