function yhat = predict_in_sample(y, s, coef)
% PREDICT_IN_SAMPLE  one-step in-sample predictions yhat_{t|t-1} for t=N+1..T
    y = y(:);
    if isstruct(coef) && isfield(coef,'coef') && ~isfield(coef,'vec') && ~isfield(coef,'c')
        % wrapper struct pattern, unwrap
        coef = coef.coef;
    end
    if ~isstruct(coef)
        error('predict_in_sample: coef must be a struct produced by select_model/unpack_coeffs.');
    end
    N = coef.N; K = coef.K;
    T = numel(y);
    M = T - N;
    if M <= 0, yhat = zeros(0,1); return; end

    yhat = zeros(M,1);
    rows = (N+1):T;
    for idx = 1:M
        t = rows(idx);
        % seasonal part
        if K > 0
            kvec = (1:K);
            TK = (2*pi/s) * (kvec * t); % 1xK
            sea = coef.alpha(:).' .* cos(TK) + coef.beta(:).' .* sin(TK);
            sea = sum(sea);
        else
            sea = 0;
        end
        val = coef.c + coef.d * t + sea;
        for i = 1:N
            val = val + coef.a(i) * y(t - i);
        end
        yhat(idx) = val;
    end
end
