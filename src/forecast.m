function yF = forecast(y, s, coef_in, H, varargin)
if nargin < 4, error('forecast requires y,s,coef_in,H'); end

% unwrap wrapper
if isstruct(coef_in) && isfield(coef_in,'coef') && ~isfield(coef_in,'vec') && ~isfield(coef_in,'c')
    coef_in = coef_in.coef;
end

[N,K,c,d,a,alpha,beta_sin] = aggressive_parse_coef(coef_in, varargin{:});

% attempt to infer missing c,d from vec if needed
if isempty(c) || isempty(d)
    if isstruct(coef_in) && isfield(coef_in,'vec') && numel(coef_in.vec) >= 2
        tmp = coef_in.vec(:); c = tmp(1); d = tmp(2);
    end
end

% final safe padding
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
