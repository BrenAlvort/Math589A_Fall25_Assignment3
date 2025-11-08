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

% Search grid
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

        % Sanity: ensure beta_vec length matches p
        if numel(beta_vec) ~= p
            % defend: if solver returned unexpected length, skip this model
            warning('Model N=%d K=%d gave beta length %d but expected %d — skipping', N, K, numel(beta_vec), p);
            Sgrid(iN, jK) = NaN;
            continue;
        end

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
            best.coef = beta_vec(:);   % temporarily store numeric vector
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

% If no candidate found, return empty but consistent struct
if isempty(best.coef)
    best.coef = struct('N', [], 'K', [], 'vec', [], 'c', [], 'd', [], 'a', [], 'alpha', [], 'beta', []);
    return;
end

% Recompute final coefficients for the chosen (N,K) to guarantee correct size
N = best.N;
K = best.K;
try
    [A_final, b_final] = build_design(y, s, N, K);
    [beta_final, RSS_final, ~] = qr_householder_solve(A_final, b_final);
catch ME
    error('select_model: failed to recompute final fit for chosen N=%d K=%d: %s', N, K, ME.message);
end

% Validate length
p_expected = 2 + N + 2*K;
if length(beta_final) ~= p_expected
    error('select_model: final coefficient length mismatch: got %d expected %d', length(beta_final), p_expected);
end

tmpvec = beta_final(:);

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

% Pack final coefficient struct (guaranteed consistent)
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

% Also update RSS/p/M for final
best.RSS = RSS_final;
best.p = p_expected;
best.M = T - N;

end

