function yF = forecast(y, s, coef_in, H, varargin)
% Robust forecast accepting many coef formats

if nargin < 4, error('forecast requires y,s,coef_in,H'); end

% parse coef
[ N, K, c, d, a, alpha, beta_sin ] = parse_coef_forcast_helper(coef_in, varargin{:});

% ensure shapes
a = reshape(a, [], 1);
alpha = reshape(alpha, [], 1);
beta_sin = reshape(beta_sin, [], 1);
if numel(a) < N, a = [a; zeros(N - numel(a),1)]; end
if numel(alpha) < K, alpha = [alpha; zeros(K - numel(alpha),1)]; end
if numel(beta_sin) < K, beta_sin = [beta_sin; zeros(K - numel(beta_sin),1)]; end

T = length(y);
y_ext = [y(:); zeros(H,1)];
for h = 1:H
    t = T + h;
    val = c + d * t;
    for i = 1:N
        idx = t - i;
        val = val + a(i) * y_ext(idx);
    end
    for k = 1:K
        val = val + alpha(k) * cos(2*pi*k*t/s) + beta_sin(k) * sin(2*pi*k*t/s);
    end
    y_ext(T + h) = val;
end

yF = y_ext(T+1:T+H);
end

function [N,K,c,d,a,alpha,beta_sin] = parse_coef_forcast_helper(coef_in, varargin)
% same logic as predict_in_sample.parse_coef but as a separate helper

% defaults
N = 0; K = 0; c = []; d = []; a = zeros(0,1); alpha = zeros(0,1); beta_sin = zeros(0,1);

% unwrap wrapper
if isstruct(coef_in) && isfield(coef_in,'coef') && ~isfield(coef_in,'vec') && ~isfield(coef_in,'c')
    coef_in = coef_in.coef;
end

if isnumeric(coef_in)
    if numel(varargin) < 2, error('forecast: numeric coef requires N and K'); end
    N = varargin{1}; K = varargin{2};
    tmp = coef_in(:);
    if numel(tmp) ~= 2 + N + 2*K, error('forecast: coef vec length mismatch'); end
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

    if isfield(coef_in,'c') && isfield(coef_in,'d')
        c = coef_in.c; d = coef_in.d;
    end

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

    if isfield(coef_in,'a'), a = coef_in.a(:); N = numel(a); end
    if isfield(coef_in,'alpha'), alpha = coef_in.alpha(:); K = numel(alpha); end
    if isfield(coef_in,'beta'), beta_sin = coef_in.beta(:); K = max(K, numel(beta_sin)); end

    % try to find the largest numeric vector field as candidate vec
    f = fieldnames(coef_in);
    bestlen = 0; bestvec = [];
    for i=1:numel(f)
        val = coef_in.(f{i});
        if isnumeric(val) && numel(val) > bestlen
            bestlen = numel(val);
            bestvec = val(:);
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
    error('forecast: coef struct must include either fields c,d or vec.');
end
end
