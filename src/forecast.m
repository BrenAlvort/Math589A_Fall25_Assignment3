function yF = forecast(y, s, coef_in, H, varargin)
% FORECAST  h-step ahead forecasts using fixed linear difference equation + deterministic seasonality
% Usage:
%   yF = forecast(y, s, coef_struct, H)
%   yF = forecast(y, s, coef_vec, H, N, K)

if nargin < 4, error('forecast requires y,s,coef_in,H'); end

% Unwrap wrapper if needed
if isstruct(coef_in) && isfield(coef_in,'coef') && ~isfield(coef_in,'vec') && ~isfield(coef_in,'c')
    coef_in = coef_in.coef;
end

% parse similar to predict_in_sample
if isnumeric(coef_in)
    if numel(varargin) < 2, error('forecast: when coef is numeric supply N and K'); end
    N = varargin{1}; K = varargin{2};
    tmp = coef_in(:);
    c = tmp(1); d = tmp(2);
    if N>0, a = tmp(3:2+N); else a = zeros(0,1); end
    if K>0
        alpha = tmp(2+N+1:2+N+K);
        beta_sin = tmp(2+N+K+1:2+N+2*K);
    else
        alpha = zeros(0,1); beta_sin = zeros(0,1);
    end

elseif isstruct(coef_in)
    if isfield(coef_in,'N') && isfield(coef_in,'K')
        N = coef_in.N; K = coef_in.K;
    else
        if isfield(coef_in,'vec')
            p = numel(coef_in.vec);
            K = floor((p-2)/2);
            N = p - 2 - 2*K;
        else
            if isfield(coef_in,'a'), N = numel(coef_in.a); else N = 0; end
            if isfield(coef_in,'alpha'), K = numel(coef_in.alpha); else K = 0; end
        end
    end

    if isfield(coef_in,'c') && isfield(coef_in,'d')
        c = coef_in.c; d = coef_in.d;
    elseif isfield(coef_in,'vec')
        tmp = coef_in.vec(:); c = tmp(1); d = tmp(2);
    else
        error('forecast: coef struct must include either fields c,d or vec.');
    end

    if isfield(coef_in,'a')
        a = coef_in.a(:);
    elseif isfield(coef_in,'vec') && N>0
        tmp = coef_in.vec(:); a = tmp(3:2+N);
    else
        a = zeros(max(0,N),1);
    end

    if isfield(coef_in,'alpha')
        alpha = coef_in.alpha(:);
    elseif isfield(coef_in,'vec') && K>0
        tmp = coef_in.vec(:); alpha = tmp(2+N+1:2+N+K);
    else
        alpha = zeros(0,1);
    end

    if isfield(coef_in,'beta')
        beta_sin = coef_in.beta(:);
    elseif isfield(coef_in,'vec') && K>0
        tmp = coef_in.vec(:); beta_sin = tmp(2+N+K+1:2+N+2*K);
    else
        beta_sin = zeros(0,1);
    end
else
    error('forecast: coef_in must be numeric or struct.');
end

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
