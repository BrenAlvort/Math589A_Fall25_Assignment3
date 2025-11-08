function yF = forecast(y, s, coef_in, H, varargin)
if nargin < 4, error('forecast requires y,s,coef_in,H'); end

[N,K,c,d,a,alpha,beta_sin] = parse_coef_generic(coef_in, varargin{:});

% infer c,d from vec if needed
if (isempty(c) || isempty(d)) && isstruct(coef_in) && isfield(coef_in,'vec')
    tmp = coef_in.vec(:);
    if numel(tmp) >= 2
        c = tmp(1); d = tmp(2);
    end
end
if isempty(c) || isempty(d)
    error('forecast: coef must provide c and d (either directly or via vec).');
end

% pad parameter arrays
a = reshape(a,[],1); alpha = reshape(alpha,[],1); beta_sin = reshape(beta_sin,[],1);
if numel(a) < N, a(end+1:N,1) = 0; end
if numel(alpha) < K, alpha(end+1:K,1) = 0; end
if numel(beta_sin) < K, beta_sin(end+1:K,1) = 0; end

y = y(:);
T = numel(y);
y_ext = [y; zeros(H,1)];
for h = 1:H
    t = T + h;
    v = c + d * t;
    for i = 1:N, v = v + a(i) * y_ext(t - i); end
    for k = 1:K, v = v + alpha(k) * cos(2*pi*k*t/s) + beta_sin(k) * sin(2*pi*k*t/s); end
    y_ext(T + h) = v;
end
yF = y_ext(T+1:T+H);
end
