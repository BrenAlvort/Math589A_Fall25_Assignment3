function yF = forecast(y, s, coef_in, H)
% FORECAST  h-step ahead forecasts using coef struct returned by select_model
%   yF = forecast(y, s, coef_in, H)
% Robust: accepts either coef struct or wrapper that contains .coef.
    if nargin < 4, error('forecast requires y,s,coef,H'); end

    % unwrap wrapper if present (some callers pack best.coef inside another struct)
    if isstruct(coef_in) && isfield(coef_in,'coef') && ~isfield(coef_in,'vec') && ~isfield(coef_in,'c')
        coef = coef_in.coef;
    else
        coef = coef_in;
    end

    if ~isstruct(coef)
        error('forecast: coef must be a struct (from select_model/unpack_coeffs).');
    end

    % canonicalize fields and shapes
    if ~isfield(coef,'N'), N = 0; else N = coef.N; end
    if ~isfield(coef,'K'), K = 0; else K = coef.K; end
    c = coef.c; d = coef.d;
    a = zeros(max(0,N),1); alpha = zeros(max(0,K),1); beta_sin = zeros(max(0,K),1);
    if isfield(coef,'a'), a(1:min(numel(coef.a),N)) = coef.a(1:min(numel(coef.a),N)); end
    if isfield(coef,'alpha'), alpha(1:min(numel(coef.alpha),K)) = coef.alpha(1:min(numel(coef.alpha),K)); end
    if isfield(coef,'beta'), beta_sin(1:min(numel(coef.beta),K)) = coef.beta(1:min(numel(coef.beta),K)); end

    % ensure column vectors
    a = a(:); alpha = alpha(:); beta_sin = beta_sin(:);

    y = y(:);
    T = numel(y);

    % Precompute seasonal terms for t = T+1 ... T+H
    future_t = (T+1 : T+H).';
    if K > 0
        kvec = (1:K);
        % produce H x K matrices
        TK = (future_t * kvec) * (2*pi/s);   % HxK
        cos_mat = cos(TK);                   % HxK
        sin_mat = sin(TK);                   % HxK
    else
        cos_mat = zeros(H,0);
        sin_mat = zeros(H,0);
    end

    % Prepare extended y vector where we will append forecasts
    y_ext = [y; zeros(H,1)];

    % Forecast iteratively
    for h = 1:H
        t = T + h;                 % current absolute time
        % seasonal contribution (row h of cos_mat/sin_mat)
        if K > 0
            sea = (alpha.' * cos_mat(h,:).' ) + (beta_sin.' * sin_mat(h,:).');
            % alpha' * cos_col + beta' * sin_col gives scalar
            % above expression returns scalar but may be 1x1 array; cast:
            sea = double(sea);
        else
            sea = 0;
        end

        % Intercept + trend
        val = c + d * t + sea;

        % AR lags: build vector [y_{t-1}, y_{t-2}, ..., y_{t-N}] and dot with a
        if N > 0
            lag_idx = (t-1):-1:(t-N); % descending indices
            % For indices <= T use original data y, else use already computed y_ext
            lag_vals = arrayfun(@(idx) y_ext(idx), lag_idx).'; % column vector of length N
            % Add AR contribution
            val = val + a.' * lag_vals;   % scalar
        end

        % store forecast
        y_ext(T + h) = val;
    end

    % return H-step forecasts
    yF = y_ext(T+1 : T+H);
end
