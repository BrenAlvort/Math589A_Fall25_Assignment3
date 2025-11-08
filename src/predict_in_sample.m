function yhat = predict_in_sample(y, s, coef)
% one-step in-sample prediction using coef struct returned by select_model
% expects coef to have fields N,K,c,d,a,alpha,beta or to contain vec.

% unwrap if wrapper
if isstruct(coef) && isfield(coef,'coef') && ~isfield(coef,'vec')
    coef = coef.coef;
end

% prefer named fields; else unpack from vec
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
    % assume fields exist
    N = coef.N; K = coef.K;
    c = coef.c; d = coef.d;
    a = coef.a(:); alpha = coef.alpha(:); beta = coef.beta(:);
end

y = y(:);
T = numel(y);
yhat = zeros(T - N, 1);
rows = (N+1):T;

for i = 1:numel(rows)
    t = rows(i);
    v = c + d * t;
    for j = 1:N
        v = v + a(j) * y(t - j);
    end
    for k = 1:K
        v = v + alpha(k) * cos(2*pi*k*t/s) + beta(k) * sin(2*pi*k*t/s);
    end
    yhat(i) = v;
end
end
