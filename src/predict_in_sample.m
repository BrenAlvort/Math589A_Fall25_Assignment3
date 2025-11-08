function yhat = predict_in_sample(y, s, coef_in, varargin)
% PREDICT_IN_SAMPLE  one-step-ahead in-sample predictions yhat for t = N+1..T
% Usage:
%   yhat = predict_in_sample(y, s, coef_struct)          % coef_struct ideally has fields N,K,c,d,a,alpha,beta or vec
%   yhat = predict_in_sample(y, s, coef_vec, N, K)       % raw numeric vector + N and K
%
T = length(y);

% Parse coef_in into canonical variables: N,K,c,d,a,alpha,beta
if isnumeric(coef_in)
    % numeric vector provided: require N,K in varargin
    if numel(varargin) < 2
        error('predict_in_sample: when coef is numeric you must also provide N and K as additional arguments.');
    end
    N = varargin{1};
    K = varargin{2};
    tmp = coef_in(:);
    if length(tmp) ~= 2 + N + 2*K
        error('predict_in_sample: length(coef_vec) mismatch with supplied N,K.');
    end
    c = tmp(1); d = tmp(2);
    if N>0, a = tmp(3:2+N); else a = zeros(0,1); end
    if K>0
        alpha = tmp(2+N+1:2+N+K);
        beta_sin = tmp(2+N+K+1:2+N+2*K);
    else
        alpha = zeros(0,1); beta_sin = zeros(0,1);
    end

elseif isstruct(coef_in)
    % If N/K are present use them, otherwise try to infer
    if isfield(coef_in,'N') && isfield(coef_in,'K')
        N = coef_in.N; K = coef_in.K;
    else
        % infer from presence of 'a','alpha','beta' or from vec length
        if isfield(coef_in,'a'), N = numel(coef_in.a); else N = []; end
        if isfield(coef_in,'alpha'), K = numel(coef_in.alpha); elseif isfield(coef_in,'beta'), K = numel(coef_in.beta); else K = []; end
        if isempty(N) || isempty(K)
            if isfield(coef_in,'vec')
                p = numel(coef_in.vec);
                % choose K = floor((p-2)/2), N = p-2-2K  (prefer larger K)
                K = floor((p-2)/2);
                N = p - 2 - 2*K;
            else
                % default to zero memory/seasonality
                if isempty(N), N = 0; end
                if isempty(K), K = 0; end
            end
        end
    end

    % c and d
    if isfield(coef_in,'c') && isfield(coef_in,'d')
        c = coef_in.c; d = coef_in.d;
    elseif isfield(coef_in,'vec')
        tmp = coef_in.vec(:); c = tmp(1); d = tmp(2);
    else
        error('predict_in_sample: coef struct must include either fields c,d or vec.');
    end

    % a
    if isfield(coef_in,'a')
        a = coef_in.a(:);
    elseif isfield(coef_in,'vec') && N>0
        tmp = coef_in.vec(:); a = tmp(3:2+N);
    else
        a = zeros(max(0,N),1);
    end

    % seasonal
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
    error('predict_in_sample: coef_in must be a struct or numeric vector.');
end

% final sanity: ensure sizes are consistent
if numel(a) ~= N, a = reshape(a,[],1); end
if numel(alpha) ~= K, alpha = reshape(alpha,[],1); end
if numel(beta_sin) ~= K, beta_sin = reshape(beta_sin,[],1); end

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
