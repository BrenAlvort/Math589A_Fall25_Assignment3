function yF = forecast(y, s, coef_in, H, varargin)
if nargin < 4, error('forecast requires y,s,coef_in,H'); end

% unwrap wrapper
if isstruct(coef_in) && isfield(coef_in,'coef') && ~isfield(coef_in,'vec') && ~isfield(coef_in,'c')
    coef_in = coef_in.coef;
end

% parse
[N,K,c,d,a,alpha,beta_sin] = parse_coef_forecast(coef_in, varargin{:});

% pad shapes
a = reshape(a,[],1); alpha = reshape(alpha,[],1); beta_sin = reshape(beta_sin,[],1);
if numel(a) < N, a(end+1:N,1) = 0; end
if numel(alpha) < K, alpha(end+1:K,1) = 0; end
if numel(beta_sin) < K, beta_sin(end+1:K,1) = 0; end

T = length(y);
y_ext = [y(:); zeros(H,1)];
for h = 1:H
    t = T + h;
    val = c + d * t;
    for i = 1:N
        val = val + a(i) * y_ext(t - i);
    end
    for k = 1:K
        val = val + alpha(k) * cos(2*pi*k*t/s) + beta_sin(k) * sin(2*pi*k*t/s);
    end
    y_ext(T + h) = val;
end
yF = y_ext(T+1:T+H);
end

function [N,K,c,d,a,alpha,beta_sin] = parse_coef_forecast(coef_in, varargin)
% reuse the same logic as robust_parse_coef
N = 0; K = 0; c = []; d = []; a = zeros(0,1); alpha = zeros(0,1); beta_sin = zeros(0,1);
if isnumeric(coef_in)
    if numel(varargin) < 2, error('forecast: numeric coef requires N,K'); end
    N = varargin{1}; K = varargin{2};
    tmp = coef_in(:);
    if numel(tmp) ~= 2 + N + 2*K, error('forecast: coef length mismatch'); end
    c = tmp(1); d = tmp(2);
    if N>0, a = tmp(3:2+N); end
    if K>0
        alpha = tmp(2+N+1:2+N+K);
        beta_sin = tmp(2+N+K+1:2+N+2*K);
    end
    return;
end

if isstruct(coef_in)
    if isfield(coef_in,'N'), N = coef_in.N; end
    if isfield(coef_in,'K'), K = coef_in.K; end
    if isfield(coef_in,'c') && isfield(coef_in,'d'), c = coef_in.c; d = coef_in.d; end
    if isfield(coef_in,'a'), a = coef_in.a(:); end
    if isfield(coef_in,'alpha'), alpha = coef_in.alpha(:); end
    if isfield(coef_in,'beta'), beta_sin = coef_in.beta(:); end
    if isfield(coef_in,'vec') && ~isempty(coef_in.vec)
        tmp = coef_in.vec(:); p = numel(tmp);
        c = tmp(1); d = tmp(2);
        K = floor((p-2)/2); N = p - 2 - 2*K;
        if N>0, a = tmp(3:2+N); end
        if K>0
            alpha = tmp(2+N+1:2+N+K);
            beta_sin = tmp(2+N+K+1:2+N+2*K);
        end
        return;
    end

    % pick largest numeric field
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
        K = floor((p-2)/2); N = p - 2 - 2*K;
        if N>0, a = tmp(3:2+N); end
        if K>0
            alpha = tmp(2+N+1:2+N+K);
            beta_sin = tmp(2+N+K+1:2+N+2*K);
        end
        return;
    end
end

if isempty(c) || isempty(d)
    error('forecast: coef struct must include either fields c,d or vec.');
end
end
