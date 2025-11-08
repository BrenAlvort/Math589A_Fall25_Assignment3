function yF = forecast(y, s, coef, H, varargin)
% h-step ahead forecasts using coef struct (must have c,d,a,alpha,beta or vec)
if isstruct(coef) && isfield(coef,'coef') && ~isfield(coef,'vec')
    coef = coef.coef;
end

% parse
if isstruct(coef) && isfield(coef,'vec') && ~isempty(coef.vec)
    vec = coef.vec(:);
    N = coef.N; K = coef.K;
    c = vec(1); d = vec(2);
    a = zeros(max(0,N),1); alpha = zeros(max(0,K),1); beta = zeros(max(0,K),1);
    idx = 3;
    for i = 1:N, a(i) = vec(idx); idx = idx + 1; end
    for k = 1:K, alpha(k) = vec(idx); idx = idx + 1; end
    for k = 1:K, beta(k) = vec(idx); idx = idx + 1; end
else
    N = coef.N; K = coef.K;
    c = coef.c; d = coef.d;
    a = coef.a(:); alpha = coef.alpha(:); beta = coef.beta(:);
end

y = y(:);
T = numel(y);
y_ext = [y; zeros(H,1)];

for h = 1:H
    t = T + h;
    v = c + d * t;
    for i = 1:N
        v = v + a(i) * y_ext(t - i);
    end
    for k = 1:K
        v = v + alpha(k) * cos(2*pi*k*t/s) + beta(k) * sin(2*pi*k*t/s);
    end
    y_ext(T + h) = v;
end

yF = y_ext(T+1 : T+H);
end
