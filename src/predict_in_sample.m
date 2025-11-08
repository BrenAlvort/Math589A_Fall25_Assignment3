function yhat = predict_in_sample(y, s, coef_in, varargin)
% robust in-sample one-step predictions; uses parse_coef_generic

T = numel(y);

[N,K,c,d,a,alpha,beta_sin] = parse_coef_generic(coef_in, varargin{:});

% If c or d empty try to infer from vec (if present)
if (isempty(c) || isempty(d)) && isstruct(coef_in) && isfield(coef_in,'vec')
    tmp = coef_in.vec(:);
    if numel(tmp) >= 2
        c = tmp(1); d = tmp(2);
    end
end
if isempty(c) || isempty(d)
    error('predict_in_sample: coef must provide c and d (either directly or via vec).');
end

% pad parameter arrays
a = reshape(a,[],1); alpha = reshape(alpha,[],1); beta_sin = reshape(beta_sin,[],1);
if numel(a) < N, a(end+1:N,1) = 0; end
if numel(alpha) < K, alpha(end+1:K,1) = 0; end
if numel(beta_sin) < K, beta_sin(end+1:K,1) = 0; end

y = y(:);
yhat = zeros(T - N, 1);
rows = (N+1):T;
for idx = 1:numel(rows)
    t = rows(idx);
    v = c + d * t;
    for i = 1:N, v = v + a(i) * y(t - i); end
    for k = 1:K, v = v + alpha(k) * cos(2*pi*k*t/s) + beta_sin(k) * sin(2*pi*k*t/s); end
    yhat(idx) = v;
end
end
