function yF = forecast(y, s, coef_in, H)
% FORECAST  h-step ahead forecasts using fixed linear difference equation + deterministic seasonality
%   yF = forecast(y, s, coef_in, H)
%
% coef_in must be a struct (see select_model), H is forecast horizon.

if nargin < 4
    error('forecast requires four inputs: y, s, coef_in, H');
end

if isstruct(coef_in)
    if isfield(coef_in,'N') && isfield(coef_in,'K')
        N = coef_in.N;
        K = coef_in.K;
    else
        error('forecast: coef struct must include fields N and K.');
    end

    if isfield(coef_in,'c') && isfield(coef_in,'d')
        c = coef_in.c;
        d = coef_in.d;
    elseif isfield(coef_in,'vec')
        tmp = coef_in.vec(:);
        c = tmp(1);
        d = tmp(2);
    else
        error('forecast: coef struct must include either {c,d} or vec.');
    end

    if isfield(coef_in,'a')
        a = coef_in.a(:);
    elseif isfield(coef_in,'vec') && N>0
        tmp = coef_in.vec(:);
        a = tmp(3:2+N);
    else
        a = zeros(max(0,N),1);
    end

    if isfield(coef_in,'alpha')
        alpha = coef_in.alpha(:);
    elseif isfield(coef_in,'vec') && K>0
        tmp = coef_in.vec(:);
        alpha = tmp(2+N+1 : 2+N+K);
    else
        alpha = zeros(0,1);
    end

    if isfield(coef_in,'beta')
        beta_sin = coef_in.beta(:);
    elseif isfield(coef_in,'vec') && K>0
        tmp = coef_in.vec(:);
        beta_sin = tmp(2+N+K+1 : 2+N+2*K);
    else
        beta_sin = zeros(0,1);
    end
else
    error('forecast: coef_in must be a struct (returned by select_model).');
end

T = length(y);

% Extended series (past + forecast slots)
y_ext = [y(:); zeros(H,1)];

for h = 1:H
    t = T + h;   % absolute time index of the forecast step
    val = c + d * t;
    for i = 1:N
        idx = t - i; % index into y_ext (works both for in-sample and forecast lags)
        val = val + a(i) * y_ext(idx);
    end
    for k = 1:K
        val = val + alpha(k) * cos(2*pi*k*t/s) + beta_sin(k) * sin(2*pi*k*t/s);
    end
    y_ext(T + h) = val;
end

yF = y_ext(T+1 : T+H);
end
