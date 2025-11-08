function best = select_model(y, s, Ngrid, Kgrid, criterion)
% SELECT_MODEL search grid, return best struct with named fields in best.coef
if nargin < 5, criterion = 'bic'; end
T = numel(y);

best.score = inf;
best.N = NaN;
best.K = NaN;
best.coef = [];
Sgrid = nan(numel(Ngrid), numel(Kgrid));

for i = 1:numel(Ngrid)
    N = Ngrid(i);
    for j = 1:numel(Kgrid)
        K = Kgrid(j);
        try
            [A,b,meta] = build_design(y, s, N, K);
        catch
            Sgrid(i,j) = NaN;
            continue;
        end

        beta = qr_solve_dense(A,b);
        res = A*beta - b;
        RSS = sum(res.^2);

        p = 2 + N + 2*K;
        M = meta.rows;

        if strcmpi(criterion,'bic')
            S = M * log(RSS / M) + p * log(M);
        else
            S = M * log(RSS / M) + 2 * p;
        end

        Sgrid(i,j) = S;

        if ~isnan(S) && S < best.score
            best.score = S;
            best.N = N;
            best.K = K;
            best.coef = beta(:);  % temporarily numeric
            best.RSS = RSS;
            best.M = M;
            best.p = p;
        end
    end
end

best.S = Sgrid;
best.Ngrid = Ngrid;
best.Kgrid = Kgrid;

% If nothing found return consistent empty coef
if isempty(best.coef)
    best.coef = struct('N', [], 'K', [], 'vec', [], 'c', [], 'd', [], 'a', [], 'alpha', [], 'beta', []);
    return;
end

% Recompute final fit with chosen N,K and pack named fields exactly
N = best.N; K = best.K;
[A_final, b_final, meta] = build_design(y, s, N, K);
beta_final = qr_solve_dense(A_final, b_final);
vec = beta_final(:);

% ensure vec has expected length p
p_expected = 2 + N + 2*K;
if numel(vec) < p_expected
    vec = [vec; zeros(p_expected - numel(vec),1)];
elseif numel(vec) > p_expected
    vec = vec(1:p_expected);
end

% Unpack
c = vec(1);
d = vec(2);

a = zeros(max(0,N),1);
alpha = zeros(max(0,K),1);
beta_sin = zeros(max(0,K),1);

idx = 3;
for i = 1:N
    a(i) = vec(idx); idx = idx + 1;
end
for k = 1:K
    alpha(k) = vec(idx); idx = idx + 1;
end
for k = 1:K
    beta_sin(k) = vec(idx); idx = idx + 1;
end

% ensure col vectors
a = a(:); alpha = alpha(:); beta_sin = beta_sin(:);

best.coef = struct('N', N, 'K', K, 'vec', vec, 'c', c, 'd', d, 'a', a, 'alpha', alpha, 'beta', beta_sin);
best.RSS = sum((A_final*vec - b_final).^2);
best.p = p_expected;
best.M = meta.rows;
end
