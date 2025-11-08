function best = select_model(y, s, Ngrid, Kgrid, criterion)
if nargin < 5, criterion = 'bic'; end
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
            best.coef = beta_vec(:);   % temporarily store
            best.RSS = RSS;
            best.M = M;
            best.p = p;
        end
    end
end

best.S = Sgrid;
best.Ngrid = Ngrid;
best.Kgrid = Kgrid;

% if no candidate found
if isempty(best.coef)
    best.coef = struct('N', [], 'K', [], 'vec', [], 'c', [], 'd', [], 'a', [], 'alpha', [], 'beta', []);
    return;
end

% Recompute the final fit for the chosen N,K to ensure consistency
N = best.N; K = best.K;
[A_final, b_final] = build_design(y, s, N, K);
[beta_final, RSS_final, ~] = qr_householder_solve(A_final, b_final);

p_expected = 2 + N + 2*K;
if numel(beta_final) ~= p_expected
    % Diagnostic print then attempt to continue safely by padding/truncating
    fprintf('[select_model] WARNING: final beta length %d != expected %d. Will pad/truncate.\n', numel(beta_final), p_expected);
end

vec = beta_final(:);
% Pad or truncate vec to p_expected safely
if numel(vec) < p_expected
    vec = [vec; zeros(p_expected - numel(vec), 1)];
elseif numel(vec) > p_expected
    vec = vec(1:p_expected);
end

% Unpack robustly (safe indices)
c = vec(1);
d = vec(2);

a = zeros(max(0,N),1);
alpha = zeros(max(0,K),1);
beta_sin = zeros(max(0,K),1);

if N > 0
    s1 = 3; s2 = 2 + N;
    rng = s1:min(s2, numel(vec));
    a(1:numel(rng)) = vec(rng);
end
if K > 0
    sa1 = 2 + N + 1; sa2 = 2 + N + K;
    rngA = sa1:min(sa2, numel(vec));
    alpha(1:numel(rngA)) = vec(rngA);

    sb1 = 2 + N + K + 1; sb2 = 2 + N + 2*K;
    rngB = sb1:min(sb2, numel(vec));
    beta_sin(1:numel(rngB)) = vec(rngB);
end

% Ensure column shape
a = reshape(a,[],1);
alpha = reshape(alpha,[],1);
beta_sin = reshape(beta_sin,[],1);

% Pack final struct with all expected named fields
best.coef = struct('N', N, 'K', K, 'vec', vec, 'c', c, 'd', d, 'a', a, 'alpha', alpha, 'beta', beta_sin);
best.RSS = RSS_final;
best.p = p_expected;
best.M = T - N;

% Diagnostic print (small)
fprintf('[select_model] final pack: N=%d K=%d len(vec)=%d lengths: a=%d alpha=%d beta=%d\n', ...
    best.coef.N, best.coef.K, numel(best.coef.vec), numel(best.coef.a), numel(best.coef.alpha), numel(best.coef.beta));
end
