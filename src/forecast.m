function yF = forecast(y, s, coef, H)
% FORECAST  h-step ahead forecasting using fitted coef struct
    if nargin < 4, error('forecast requires y,s,coef,H'); end
    if isstruct(coef) && isfield(coef,'coef') && ~isfield(coef,'vec') && ~isfield(coef,'c')
        coef = coef.coef;
    end
    if ~isstruct(coef)
        error('forecast: coef must be a struct, e.g. from select_model');
    end

    y = y(:);
    T = numel(y);
    N = coef.N; K = coef.K;
    % ensure vectors present
    a = coef.a(:); alpha = coef.alpha(:); beta_sin = coef.beta(:);
    if numel(a) < N, a(end+1:N) = 0; end
    if numel(alpha) < K, alpha(end+1:K) = 0; end
    if numel(beta_sin) < K, beta_sin(end+1:K) = 0; end

    y_ext = [y; zeros(H,1)];
    for h = 1:H
        t = T + h;
        % deterministic seasonal
        sea = 0;
        for k = 1:K
            sea = sea + alpha(k)*cos(2*pi*k*t/s) + beta_sin(k)*sin(2*pi*k*t/s);
        end
        val = coef.c + coef.d * t + sea;
        for i = 1:N
            idx = t - i;
            if idx <= T
                val = val + a(i) * y(idx);
            else
                % use previously computed forecast
                val = val + a(i) * y_ext(idx);
            end
        end
        y_ext(T + h) = val;
    end
    yF = y_ext(T+1:T+H);
end
