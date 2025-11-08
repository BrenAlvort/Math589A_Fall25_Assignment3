function best = select_model(y, s, Ngrid, Kgrid, criterion)
% SELECT_MODEL Search over grids of (N,K) and pick best by criterion (e.g., 'bic')
if nargin < 5
    criterion = 'bic';
end

T = length(y);
best.score = inf;
best.N = NaN;
best.K = NaN;
best.coef = [];
Sgrid = nan(length(Ngrid), length(Kgrid));

% grid search
for iN = 1:length(Ngrid)
    N = Ngrid(iN);
    for jK = 1:length(Kgrid)
        K = Kgrid(jK);
        try
            [A, b] = build_design(y, s, N, K);
        catch ME
            Sgrid(iN, jK) = NaN;
            warning('Skipping N=%d K=%d: %s', N, K, ME.message);
            continue;
        end

        [beta_vec, RSS, ~] = qr_householder_solve(A, b);

        % expected parameter count
        p = 2 + N + 2*K;
        if numel(beta_vec) ~= p
            warning('Model N=%d K=%d gave beta length %d but expected %d — skipping', N, K, numel(beta_vec), p);
            Sgrid(iN, jK) = NaN;
            continue;
        end

        M = T - N;
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

        if ~isnan(S) && S < best.score
            best.score = S;
            best.N = N;
            best.K = K;
            best.coef = beta_vec(:);   % temp store numeric vector
            best.RSS = RSS;
            best.M = M;
            best.p = p;
        end
    end
end

% store diagnostics
best.S = Sgrid;
best.Ngrid = Ngrid;
best.Kgrid = Kgrid;

% if no model found, return empty consistent struct
if isempty(best.coef)
    best.coef = struct('N', [], 'K', [], 'vec', [], 'c', [], 'd', []);
    return;
end

% recompute final fit for chosen (N,K) to guarantee consistency
N = best.N; K = best.K;
[A_final, b_final] = build_design(y, s, N, K);
[beta_final, RSS_final, ~] = qr_householder_solve(A_final, b_final);

p_expected = 2 + N + 2*K;
if length(beta_final) ~= p_expected
    error('select_model: final coefficient length mismatch: got %d expected %d', length(beta_final), p_expected);
end

tmpvec = beta_final(:);
c = tmpvec(1);
d = tmpvec(2);

% pack a minimal, reliable coef struct (N,K,vec,c,d). downstream code will derive a/alpha/beta from vec
best.coef = struct('N', N, 'K', K, 'vec', tmpvec, 'c', c, 'd', d);

% update final metadata
best.RSS = RSS_final;
best.p = p_expected;
best.M = T - N;
end
