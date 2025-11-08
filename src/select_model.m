function best = select_model(y, s, Ngrid, Kgrid, criterion)
if nargin < 5, criterion = 'bic'; end
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
        catch
            Sgrid(iN, jK) = NaN;
            continue;
        end
        [beta_vec, RSS, ~] = qr_householder_solve(A, b);

        p = 2 + N + 2*K;
        if numel(beta_vec) ~= p
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
                error('Unknown criterion');
        end
        Sgrid(iN, jK) = S;

        if ~isnan(S) && S < best.score
            best.score = S;
            best.N = N;
            best.K = K;
            best.coef = beta_vec(:);   % temporary numeric vector
            best.RSS = RSS;
            best.M = M;
            best.p = p;
        end
    end
end

best.S = Sgrid;
best.Ngrid = Ngrid;
best.Kgrid = Kgrid;

% if nothing found, return a consistent empty struct
if isempty(best.coef)
    best.coef = struct('N', [], 'K', [], 'vec', [], 'c', [], 'd', [], 'a', [], 'alpha', [], 'beta', []);
    return;
end

% Recompute final solution for chosen N,K to guarantee consistency
N = best.N; K = best.K;
[A_final, b_final] = build_design(y, s, N, K);
[beta_final, RSS_final, ~] = qr_householder_solve(A_final, b_final);

p_expected = 2 + N + 2*K;
if length(beta_final) ~= p_expected
    error('select_model: unexpected final coefficient length');
end

vec = beta_final(:);         % guaranteed column vector
c = vec(1); d = vec(2);

% safe, bounds-checked slicing for a, alpha, beta
a = zeros(max(0,N),1);
alpha = zeros(max(0,K),1);
beta_sin = zeros(max(0,K),1);

if N > 0
    s1 = 3; s2 = 2 + N;
    if s2 <= numel(vec)
        a(1:min(N, s2-s1+1)) = vec(s1:min(s2,numel(vec)));
    end
end
if K > 0
    sa1 = 2 + N + 1; sa2 = 2 + N + K;
    sb1 = 2 + N + K + 1; sb2 = 2 + N + 2*K;
    if sa1 <= numel(vec)
        alpha(1:min(K, max(0, min(sa2,numel(vec)) - sa1 + 1))) = vec(sa1:min(sa2,numel(vec)));
    end
    if sb1 <= numel(vec)
        beta_sin(1:min(K, max(0, min(sb2,numel(vec)) - sb1 + 1))) = vec(sb1:min(sb2,numel(vec)));
    end
end

% final packing (guaranteed shapes)
best.coef = struct('N', N, 'K', K, 'vec', vec, 'c', c, 'd', d, 'a', a, 'alpha', alpha, 'beta', beta_sin);
best.RSS = RSS_final;
best.p = p_expected;
best.M = T - N;
end
