function best = select_model(y, s, Ngrid, Kgrid, criterion)
% SELECT_MODEL Search over grids of (N,K) and pick best by criterion (e.g., 'bic')
%   best = select_model(y,s,Ngrid,Kgrid,criterion)
%
% Outputs (fields of struct best):
%   best.N        - selected N
%   best.K        - selected K
%   best.coef     - struct with fields: N, K, vec, c, d, a, alpha, beta
%   best.score    - numeric value of the selected criterion
%   best.S        - matrix of criterion values sized length(Ngrid) x length(Kgrid)
%   best.Ngrid    - the input Ngrid
%   best.Kgrid    - the input Kgrid

if nargin < 5
    criterion = 'bic';
end

T = length(y);
best.score = inf;
best.N = NaN;
best.K = NaN;
best.coef = [];
Sgrid = nan(length(Ngrid), length(Kgrid));

for iN = 1:length(Ngrid)
    N = Ngrid(iN);
    for jK = 1:length(Kgrid)
        K = Kgrid(jK);
        % Build design and response
        try
            [A, b] = build_design(y, s, N, K);
        catch ME
            Sgrid(iN, jK) = NaN;
            warning('Skipping N=%d K=%d: %s', N, K, ME.message);
            continue;
        end

        % Solve least squares via Householder QR (user-supplied)
        [beta_vec, RSS, ~] = qr_householder_solve(A, b);

        % number of observations and parameters
        M = T - N;
        p = 2 + N + 2*K;

        % Compute selection criterion
        switch lower(criterion)
            case 'bic'
                if RSS <= 0, RSS = max(RSS, eps); end
                S = M * log(RSS / M) + p * log(M);
            case 'aic'
                if RSS <= 0, RSS = max(RSS, eps); end
                S = M * log(RSS / M) + 2 * p;
            otherwise
                error('Unknown criterion: %s', criterion);
        end

        Sgrid(iN, jK) = S;

        % Update best
        if ~isnan(S) && S < best.score
            best.score = S;
            best.N = N;
            best.K = K;
            best.coef = beta_vec(:);
            best.RSS = RSS;
            best.M = M;
            best.p = p;
        end
    end
end

% store S grid and input grids
best.S = Sgrid;
best.Ngrid = Ngrid;
best.Kgrid = Kgrid;

% Convert numeric coefficient vector into a struct that downstream code expects
if ~isempty(best.coef)
    tmpvec = best.coef(:);
    N = best.N;
    K = best.K;
    % Unpack into named parameters, robust to N or K being zero
    c = tmpvec(1);
    d = tmpvec(2);
    if N > 0
        a = tmpvec(3 : 2+N);
    else
        a = zeros(0,1);
    end
    if K > 0
        alpha = tmpvec(2+N+1 : 2+N+K);
        beta_sin = tmpvec(2+N+K+1 : 2+N+2*K);
    else
        alpha = zeros(0,1);
        beta_sin = zeros(0,1);
    end

    best.coef = struct( ...
        'N', N, ...
        'K', K, ...
        'vec', tmpvec, ...
        'c', c, ...
        'd', d, ...
        'a', a, ...
        'alpha', alpha, ...
        'beta', beta_sin ...
    );
else
    best.coef = struct('N', [], 'K', [], 'vec', [], 'c', [], 'd', [], 'a', [], 'alpha', [], 'beta', []);
end

end
