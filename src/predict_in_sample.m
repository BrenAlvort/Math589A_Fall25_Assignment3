function yhat = predict_in_sample(y, s, coef_in, varargin)
% PREDICT_IN_SAMPLE  one-step-ahead in-sample predictions yhat for t = N+1..T
% Usage:
%   yhat = predict_in_sample(y, s, coef_struct)
%   yhat = predict_in_sample(y, s, coef_vec, N, K)

T = length(y);

% Unwrap common wrappers: if the user passed the full 'best' struct that contains field 'coef',
% replace coef_in by coef_in.coef.
if isstruct(coef_in) && isfield(coef_in,'coef') && ~isfield(coef_in,'vec') && ~isfield(coef_in,'c')
    coef_in = coef_in.coef;
end

% Parse into canonical variables
if isnumeric(coef_in)
    % numeric vector: require N,K provided
    if numel(varargin) < 2
        error('predict_in_sample: when coef is numeric you must supply N and K as additional args.');
    end
    N = varargin{1}; K = varargin{2};
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
    % If coef_in contains nested 'vec' or named fields, use them
    if isfield(coef_in,'N') && isfield(coef_in,'K')
        N = coef_in.N; K = coef_in.K;
    else
        % try to infer from vec/a/alpha/beta
        if isfield(coef_in,'vec')
            p = numel(coef_in.vec);
            K = floor((p-2)/2);
            N = p - 2 - 2*K;
        elseif isfield(coef_in,'a')
            N = numel(coef_in.a);
            if isfield(coef_in,'alpha'), K = numel(coef_in.alpha); else K = 0; end
        else
            N = 0; K = 0;
        end
    end

    if isfield(coef_in,'c') && isfield(coef_in,'d')
        c = coef_in.c; d = coef_in.d;
    elseif isfield(coef_in,'vec')
        tmp = coef_in.vec(:); c = tmp(1); d = tmp(2);
    else
        error('predict_in_sample: coef struct must include either fields c,d or vec.');
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
    error('predict_in_sample: coef_in must be numeric vector or a struct.');
end

% build predictions
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
